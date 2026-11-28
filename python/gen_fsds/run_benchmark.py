"""Benchmark driver (resumable, parallel over replicates).

    cd python
    python -m gen_fsds.run_benchmark --suite quality  --out results_gen
    python -m gen_fsds.run_benchmark --suite steps    --out results_gen
    python -m gen_fsds.run_benchmark --suite covshift --out results_gen --reps 6
    python -m gen_fsds.run_benchmark --suite concept  --out results_gen --reps 6
    python -m gen_fsds.report results_gen

Each (suite, scenario, replicate) is stored as one JSON file; existing files are skipped.
"""
from __future__ import annotations

import argparse
import json
import os
import traceback

from joblib import Parallel, delayed

QUALITY_SAMPLERS = ["Marginal", "LinGauss", "RFResid", "QRF", "NSF", "CFM", "CFM-reflow1", "AmortCFM"]

# (name, kind, delta, n_per_batch, noise_scale); noise_scale < 1 means stronger feature dependence
COV_SCENARIOS = [
    ("mean_d1.0_n5000", "mean", 1.0, 5000, 1.0),
    ("scale_d1.0_n5000", "scale", 1.0, 5000, 1.0),
    ("shape_n5000", "shape", 1.0, 5000, 1.0),
    ("mean_d0.5_n5000", "mean", 0.5, 5000, 1.0),
    ("mean_d1.0_n2000", "mean", 1.0, 2000, 1.0),
    ("mean_d1.0_n5000_strongdep", "mean", 1.0, 5000, 0.35),
]
CD_SCENARIOS = [
    ("cmean_d0.6_n5000", "mean", 0.6, 5000, 1.0),
    ("cvar_d1.0_n5000", "var", 1.0, 5000, 1.0),
    ("cmean_d0.6_n2000", "mean", 0.6, 2000, 1.0),
    ("cmean_d0.6_n5000_strongdep", "mean", 0.6, 5000, 0.35),
]


def _worker(path: str, fn_name: str, kwargs: dict):
    import torch
    torch.set_num_threads(1)
    if os.path.exists(path):
        return path, "skip"
    try:
        from . import conceptdrift, covshift, quality
        from .dgp import SEM, concept_drift_task, covariate_shift_task
        p, seed, n = kwargs["p"], kwargs["seed"], kwargs["n"]
        if fn_name == "quality":
            rows = quality.run_quality(kwargs["samplers"], n, p, seed, n_feat=kwargs.get("n_feat", 8),
                                       euler_steps=kwargs.get("euler_steps"),
                                       noise_scale=kwargs.get("noise_scale", 1.0))
        elif fn_name == "covshift":
            sem = SEM(p=p, seed=seed, noise_scale=kwargs.get("noise_scale", 1.0))
            X0, X1, truth = covariate_shift_task(sem, n, seed, kwargs["kind"], kwargs["delta"])
            rows = covshift.run_covshift(X0, X1, truth, kwargs["methods"], seed,
                                         meta=dict(scenario=kwargs["scenario"], n=n, p=p))
        elif fn_name == "concept":
            sem = SEM(p=p, seed=seed, noise_scale=kwargs.get("noise_scale", 1.0))
            X0, Y0, X1, Y1, truth = concept_drift_task(sem, n, seed, kwargs["kind"], kwargs["delta"])
            rows = conceptdrift.run_concept(X0, Y0, X1, Y1, truth, kwargs["methods"], seed,
                                            meta=dict(scenario=kwargs["scenario"], n=n, p=p))
        else:
            raise ValueError(fn_name)
        with open(path, "w") as f:
            json.dump(rows, f)
        return path, "done"
    except Exception:
        return path, "ERROR\n" + traceback.format_exc()


def build_jobs(args):
    jobs = []
    out = os.path.join(args.out, args.suite)
    os.makedirs(out, exist_ok=True)
    seeds = range(args.seed0, args.seed0 + args.reps)
    if args.suite == "quality":
        for n, ns in [(n, 1.0) for n in args.quality_n] + [(5000, 0.35)]:
            for s in seeds:
                tag = f"n{n}" + ("" if ns == 1.0 else "_strongdep")
                jobs.append((os.path.join(out, f"{tag}_s{s}.json"), "quality",
                             dict(p=args.p, seed=s, n=n, samplers=QUALITY_SAMPLERS, noise_scale=ns)))
    elif args.suite == "steps":
        for s in seeds:
            jobs.append((os.path.join(out, f"s{s}.json"), "quality",
                         dict(p=args.p, seed=s, n=5000, samplers=["CFM", "CFM-reflow1", "AmortCFM"],
                              euler_steps=[1, 2, 5, 10, 20, 50], n_feat=6)))
    elif args.suite == "covshift":
        from .covshift import DEFAULT_METHODS
        methods = args.methods.split(",") if args.methods else DEFAULT_METHODS
        for name, kind, delta, n, ns in COV_SCENARIOS:
            if args.scenarios and name not in args.scenarios.split(","):
                continue
            for s in seeds:
                jobs.append((os.path.join(out, f"{name}_s{s}.json"), "covshift",
                             dict(p=args.p, seed=s, n=n, kind=kind, delta=delta, scenario=name, methods=methods,
                                  noise_scale=ns)))
    elif args.suite == "concept":
        from .conceptdrift import DEFAULT_METHODS
        methods = args.methods.split(",") if args.methods else DEFAULT_METHODS
        for name, kind, delta, n, ns in CD_SCENARIOS:
            if args.scenarios and name not in args.scenarios.split(","):
                continue
            for s in seeds:
                jobs.append((os.path.join(out, f"{name}_s{s}.json"), "concept",
                             dict(p=args.p, seed=s, n=n, kind=kind, delta=delta, scenario=name, methods=methods,
                                  noise_scale=ns)))
    return jobs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", required=True, choices=["quality", "steps", "covshift", "concept"])
    ap.add_argument("--out", default="results_gen")
    ap.add_argument("--p", type=int, default=20)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--seed0", type=int, default=0)
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--methods", default="")
    ap.add_argument("--scenarios", default="")
    ap.add_argument("--quality-n", type=int, nargs="+", default=[1000, 2000, 5000])
    args = ap.parse_args()
    jobs = build_jobs(args)
    print(f"{len(jobs)} jobs", flush=True)
    for path, status in Parallel(n_jobs=args.jobs, return_as="generator_unordered")(
            delayed(_worker)(*j) for j in jobs):
        print(os.path.basename(path), status, flush=True)


if __name__ == "__main__":
    main()
