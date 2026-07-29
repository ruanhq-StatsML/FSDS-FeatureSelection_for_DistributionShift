"""Path-order non-uniqueness for MMD / HSIC attribution on two-domain batches."""
import os
os.chdir(
    "/Users/heqiaoruan/Library/Mobile Documents/com~apple~CloudDocs/Documents/GitHub 2/Causal_Objective_Permutation_Test/Python"
)
import json
import numpy as np
from feature_codes.VIMP_mmd_benchmark import MMD
from VIMP_hsic_benchmark import HSIC
from synthetic_DGP import (
    generate_covariate_shift_dgps,
    generate_nonlinear_covariate_shift_dgp
)


def _grow_path_blocks(df1, df2, feature_path, step):
    cols = feature_path[: step + 1]
    return df1[:, cols], df2[:, cols]


def mmd_path(df1, df2, B=20, rng=None):
    """Random feature-addition paths; returns list of paths and full-batch MMD."""
    rng = np.random.default_rng() if rng is None else rng
    df1 = np.asarray(df1, dtype=float)
    df2 = np.asarray(df2, dtype=float)
    p = df1.shape[1]
    mmd_test = MMD(compute_kernel="rbf")
    mmd_orig, _ = mmd_test(df1, df2)

    paths = []
    for _ in range(B):
        feature_path = rng.permutation(p)
        curve = []
        for step in range(p):
            x1, x2 = _grow_path_blocks(df1, df2, feature_path, step)
            value, _ = mmd_test(x1, x2)
            curve.append(float(value))
        paths.append(
            {
                "feature_path": feature_path.astype(int),
                "metric_path": np.asarray(curve, dtype=float),
            }
        )
    return paths, float(mmd_orig)


def hsic_path(df1, df2, B=20, sigma=1.0, rng=None):
    """Random feature-addition paths; returns list of paths and full-batch HSIC."""
    rng = np.random.default_rng() if rng is None else rng
    df1 = np.asarray(df1, dtype=float)
    df2 = np.asarray(df2, dtype=float)
    p = df1.shape[1]
    hsic_orig = float(HSIC(df1, df2, sigma=sigma))

    paths = []
    for _ in range(B):
        feature_path = rng.permutation(p)
        curve = []
        for step in range(p):
            x1, x2 = _grow_path_blocks(df1, df2, feature_path, step)
            curve.append(float(HSIC(x1, x2, sigma=sigma)))
        paths.append(
            {
                "feature_path": feature_path.astype(int),
                "metric_path": np.asarray(curve, dtype=float),
            }
        )
    return paths, hsic_orig


def summarize_path_nonuniqueness(paths, metric_orig, atol=1e-8):
    """Quantify that random addition orders produce different attribution paths."""
    curves = np.stack([p["metric_path"] for p in paths], axis=0)
    b, p = curves.shape
    final_vals = curves[:, -1]
    return {
        "n_paths": int(b),
        "n_steps": int(p),
        "metric_orig": float(metric_orig),
        "final_mean": float(final_vals.mean()),
        "final_std": float(final_vals.std()),
        "final_min": float(final_vals.min()),
        "final_max": float(final_vals.max()),
        "final_range": float(final_vals.max() - final_vals.min()),
        "final_all_equal_orig": bool(np.allclose(final_vals, metric_orig, atol=atol)),
        "step_std_mean": float(curves.std(axis=0).mean()),
        "step_std_max": float(curves.std(axis=0).max()),
        "pairwise_curve_diff_mean": float(
            np.mean(
                [
                    np.max(np.abs(curves[i] - curves[j]))
                    for i in range(b)
                    for j in range(i + 1, b)
                ]
            )
        ),
        "n_distinct_final_values": int(len(np.unique(np.round(final_vals, 8)))),
    }


# Backward-compatible aliases from the draft notebook code.
MMD_Path = mmd_path
HSIC_path = hsic_path



#Run the nonuniqueness procedure for the covariate shift:
SCENARIOS = (
    {
        "scenario": "linear_covariate_shift",
        "generate": generate_covariate_shift_dgp,
    },
    {
        "scenario": "nonlinear_covariate_shift",
        "generate": generate_nonlinear_covariate_shift_dgp,
    },
)

RHO = 0.3
GAMMA_GRID = [0.2, 0.3]
N_REP = 3
B_PATHS = 12
MAX_N = 150
SEED_BASE = 24000
PATHS_CSV = "path_nonuniqueness_covariate_shift_paths.csv"
SUMMARY_CSV = "path_nonuniqueness_covariate_shift_summary.csv"
JSON_PATH = "path_nonuniqueness_covariate_shift_results.json"

if scenarios is None:
    scenarios = SCENARIOS
if gamma_grid is None:
    gamma_grid = GAMMA_GRID

path_rows = []
summary_rows = []
json_out = {}

for spec in scenarios:
    scenario = spec["scenario"]
    generate = spec["generate"]
    json_out[scenario] = {}
    for gamma in gamma_grid:
        gamma_key = f"gamma={float(gamma):g}"
        json_out[scenario][gamma_key] = []
        for rep in range(n_rep):
            seed = seed_base + rep + int(float(gamma) * 1000)
            X, Y, W = generate(gamma=float(gamma), rho=float(rho), seed=seed)
            df1, df2 = _split_domains(X, Y, W)
            df1, df2 = _subsample_pair(df1, df2, max_n=max_n, seed=seed + 17)
            for metric_name, runner in (
                ("mmd", lambda a, b, rng: mmd_path(a, b, B=B_paths, rng=rng)),
                ("hsic", lambda a, b, rng: hsic_path(a, b, B=B_paths, rng=rng)),
            ):
                rng = np.random.default_rng(seed + (0 if metric_name == "mmd" else 100))
                paths, metric_orig = runner(df1, df2, rng)
                summary = summarize_path_nonuniqueness(paths, metric_orig)
                summary_rows.append(
                    {
                        "scenario": scenario,
                        "metric": metric_name,
                        "rho": float(rho),
                        "gamma": float(gamma),
                        "rep": int(rep),
                        "n_paths": B_paths,
                        "n_features": int(df1.shape[1]),
                        **summary,
                    }
                )
                json_out[scenario][gamma_key].append(
                    {
                        "rep": rep,
                        "metric": metric_name,
                        **summary,
                    }
                )
                for path_id, item in enumerate(paths):
                    for step, value in enumerate(item["metric_path"]):
                        path_rows.append(
                            {
                                "scenario": scenario,
                                "metric": metric_name,
                                "rho": float(rho),
                                "gamma": float(gamma),
                                "rep": int(rep),
                                "path_id": int(path_id),
                                "step": int(step),
                                "n_features_included": int(step + 1),
                                "feature_added": int(item["feature_path"][step]),
                                "path_value": float(value),
                                "metric_orig": float(metric_orig),
                            }
                        )
paths_df = pd.DataFrame(path_rows)
summary_df = pd.DataFrame(summary_rows)
paths_df.to_csv(PATHS_CSV, index=False)
summary_df.to_csv(SUMMARY_CSV, index=False)
with open(JSON_PATH, "w", encoding="utf-8") as f:
    json.dump(json_out, f, indent=2)



#Run the nonuniqueness procedure for the concept drift:
SCENARIOS = (
    {
        "scenario": "linear_concept_drift",
        "generate": generate_concept_drift_dgp,
    },
    {
        "scenario": "nonlinear_concept_drift",
        "generate": generate_nonlinear_concept_drift_dgp,
    },
)
RHO = 0.3
DELTA_BETA_GRID = [0.3]
N_REP = 10
B_PATHS = 12
MAX_N = 300
N_SPLITS = 2
N_PATH_WORKERS = 4
SEED_BASE = 25000
PATHS_CSV = "path_nonuniqueness_concept_drift_rrisk_paths.csv"
SUMMARY_CSV = "path_nonuniqueness_concept_drift_rrisk_summary.csv"
JSON_PATH = "path_nonuniqueness_concept_drift_rrisk_results.json"
FIRST_BLOCK_REPS = 5
SCENARIO_LOOKUP = {spec["scenario"]: spec for spec in SCENARIOS}

if scenarios is None:
    scenarios = SCENARIOS
if delta_beta_grid is None:
    delta_beta_grid = DELTA_BETA_GRID
if completed is None:
    completed = _load_completed(summary_csv)

path_rows = []
if os.path.exists(paths_csv):
    path_rows = pd.read_csv(paths_csv).to_dict("records")
summary_rows = []
if os.path.exists(summary_csv):
    summary_rows = pd.read_csv(summary_csv).to_dict("records")
json_out = {}
if os.path.exists(json_path):
    with open(json_path, encoding="utf-8") as handle:
        json_out = json.load(handle)

schedule = _build_rep_schedule(n_rep=n_rep)
for scenario, rep in schedule:
    spec = SCENARIO_LOOKUP[scenario]
    generate = spec["generate"]
    json_out.setdefault(scenario, {})

    for delta_beta in delta_beta_grid:
        key = f"delta_beta={float(delta_beta):g}"
        json_out[scenario].setdefault(key, [])
        cell = (scenario, float(delta_beta), int(rep))
        if cell in completed:
            print(
                f"[path_rrisk] skip {scenario} delta_beta={delta_beta:g} rep={rep}",
                flush=True,
            )
            continue
        seed = seed_base + rep + int(float(delta_beta) * 1000)
        X, Y, W = generate(
            delta_beta=float(delta_beta), rho=float(rho), seed=seed
        )
        X1, Y1, X2, Y2 = _split_domains(X, Y, W)
        X1, Y1, X2, Y2 = _subsample_pair(
            X1, Y1, X2, Y2, max_n=max_n, seed=seed + 17
        )
        W_sub = np.concatenate(
            [
                np.zeros(len(Y1), dtype=int),
                np.ones(len(Y2), dtype=int),
            ]
        )
        Y_sub = np.concatenate([Y1, Y2])
        rng = np.random.default_rng(seed + 200)
        paths, rrisk_orig = rrisk_path(
            X1,
            X2,
            Y_sub,
            W_sub,
            B=B_paths,
            rng=rng,
            seed=seed + 400,
            n_splits=N_SPLITS,
            n_workers=N_PATH_WORKERS,
            fast=True,
        )
        summary = summarize_path_nonuniqueness(paths, rrisk_orig)
        summary_rows.append(
            {
                "scenario": scenario,
                "metric": "rrisk",
                "rho": float(rho),
                "delta_beta": float(delta_beta),
                "rep": int(rep),
                "n_paths": B_paths,
                "n_features": int(X1.shape[1]),
                **summary,
            }
        )
        json_out[scenario][key].append(
            {
                "rep": rep,
                "metric": "rrisk",
                **summary,
            }
        )
        for path_id, item in enumerate(paths):
            for step, value in enumerate(item["metric_path"]):
                path_rows.append(
                    {
                        "scenario": scenario,
                        "metric": "rrisk",
                        "rho": float(rho),
                        "delta_beta": float(delta_beta),
                        "rep": int(rep),
                        "path_id": int(path_id),
                        "step": int(step),
                        "n_features_included": int(step + 1),
                        "feature_added": int(item["feature_path"][step]),
                        "path_value": float(value),
                        "metric_orig": float(rrisk_orig),
                    }
                )
        completed.add(cell)
        pd.DataFrame(path_rows).to_csv(paths_csv, index=False)
        pd.DataFrame(summary_rows).to_csv(summary_csv, index=False)
        with open(json_path, "w", encoding="utf-8") as handle:
            json.dump(json_out, handle, indent=2)
paths_df = pd.DataFrame(path_rows)
summary_df = pd.DataFrame(summary_rows)
paths_df.to_csv(paths_csv, index=False)
summary_df.to_csv(summary_csv, index=False)
with open(json_path, "w", encoding="utf-8") as handle:
    json.dump(json_out, handle, indent=2)
return paths_df, summary_df

























'''

DATEDIFF(A.cohort_date,  A.login_date, 'Days') <= 7
DATE_ADD(A.cohort_date,  A.login_date, 'Days') >= 7
DATE_ADD(A.cohort_date,  A.login_date, 'Days') <= 7
DATE_ADD(A.cohort_date,  A.login_date, 'Days') >= 7

'''













































