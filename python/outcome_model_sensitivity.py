"""Outcome-model sensitivity: PermuCATE / DRPerm with RF / XGB / MLP / Ridge. 
Leveraged from the model registry.

Reports selection F1 and selection_AUC across synthetic linear/nonlinear
concept-drift & covariate-shift, plus WhyShift, ChronoBerg, and 8 tabular datasets.
The originally code is reformulated by cursor.
"""
from __future__ import annotations
import argparse
import gc
import glob
import itertools
import os
import re
import sys
import numpy as np
import pandas as pd
from benchmark_config import MODEL_REGISTRY
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from adversarial_perturbation_distribution_shift_whole import impose_adv_shift
from fetch_tabular_openml_datasets import DATASET_LIST
from realdata_vimp_benchmark import (
    append_and_save,
    benchmark_permucate_drperm,
    benchmark_rows_from_result,
    load_completed_estimator_ids,
)
from synthetic_DGP import (
    synthetic_covariate_shift_nonlinear_interaction,
    synthetic_injection_nonlinear_interaction,
    synthetic_injection_nonlinear_interaction,
    generate_concept_drift_dgp,
    generate_covariate_shift_dgp,
    generate_nonlinear_concept_drift_dgp,
    generate_nonlinear_covariate_shift_dgp,
    SHIFTED_FEATURES
    )

SCORES_CSV = "outcomemodel_sensitivity_scores.csv"
METRICS_CSV = "outcomemodel_sensitivity_metrics.csv"
SUMMARY_CSV = "outcomemodel_sensitivity_summary.csv"

#The specifications for the models:
ESTIMATOR_BUNDLES = {
    "rf": {
        "model_m": "rf_regressor",
        "model_tau": "rf_regressor",
        "model_nu": "rf_regressor",
        "model_po": "rf_regressor",
        "model_e": "logistic_classifier",
    },
    "xgb": {
        "model_m": "xgb_regressor",
        "model_tau": "xgb_regressor",
        "model_nu": "xgb_regressor",
        "model_po": "xgb_regressor",
        "model_e": "logistic_classifier",
    },
    "mlp": {
        "model_m": "mlp_regressor",
        "model_tau": "mlp_regressor",
        "model_nu": "mlp_regressor",
        "model_po": "mlp_regressor",
        "model_e": "logistic_classifier",
    },
    "ridge": {
        "model_m": "ridge_regressor",
        "model_tau": "ridge_regressor",
        "model_nu": "ridge_regressor",
        "model_po": "ridge_regressor",
        "model_e": "logistic_classifier",
    },
}


#The settings of the synthetic dataset:
SYNTHETIC_SETTINGS = {
    "linear_concept_drift": {
        "fn": generate_concept_drift_dgp,
        "kwargs": {"delta_beta": 0.5, "rho": 0.3},
        "shift_type": "linear_concept_drift",
    },
    "linear_covariate_shift": {
        "fn": generate_covariate_shift_dgp,
        "kwargs": {"gamma": 0.5, "rho": 0.1},
        "shift_type": "linear_covariate_shift",
    },
    "nonlinear_concept_drift": {
        "fn": generate_nonlinear_concept_drift_dgp,
        "kwargs": {"delta_beta": 0.5, "rho": 0.3},
        "shift_type": "nonlinear_concept_drift",
    },
    "nonlinear_covariate_shift": {
        "fn": generate_nonlinear_covariate_shift_dgp,
        "kwargs": {"gamma": 0.5, "rho": 0.3},
        "shift_type": "nonlinear_covariate_shift",
    },
}

#The list of the tabular datasets, including the whyshift, 8 real-world datasets.
TABULAR_DIR = os.environ.get("TABULAR_DATA_DIR", "real_data/tabular")
TABULAR_SHIFTS = ("linear", "nonlinear_interaction", "nonlinear_covariate_shift")
SUBSAMPLE_N = 1500
N_SHIFT_PROP = 0.5
WHYSHIFT_STATES = [
    "AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "FL",
    "GA", "HI", "ID", "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD",
    "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ", "NM",
    "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI", "SC", "SD", "TN",
    "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY", "PR",
]
PAIR_PATTERN = re.compile(
    r"^(?P<s1>[A-Z]{2})_(?P<s2>[A-Z]{2})_subsample_(?P<n>\d+)_pairdf\.csv$"
)
#Chronoberg datasets:
CHRONOBERG_EMB_DIR = "real_data/yearly_extracted_batch"
CHRONOBERG_EMB_SUFFIX = "sentence_chronoberg_processed_Emb_gemma_embedding2.npy"
_CHRONO_N = int(os.environ.get("OMS_CHRONO_N_YEARS", "10"))
CHRONOBERG_YEARS = np.round(np.linspace(1750, 1970, _CHRONO_N)).astype(int)
_CHRONO_ADV = os.environ.get("OMS_CHRONO_ADV", "miFGSM,rFGSM,jitter")
CHRONOBERG_ADV = tuple(a.strip() for a in _CHRONO_ADV.split(",") if a.strip())


def _xyw_to_batches(X, Y, W):
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float).ravel()
    W = np.asarray(W, dtype=int).ravel()
    df1 = np.hstack([X[W == 0], Y[W == 0].reshape(-1, 1)])
    df2 = np.hstack([X[W == 1], Y[W == 1].reshape(-1, 1)])
    return df1, df2


def run_individual_version(
    scenario,
    dataset_id,
    shift_type,
    df1,
    df2,
    feature_ind,
    estimator,
    rep,
    seed,
    n_perm,
    completed,
):
    key = (scenario, dataset_id, shift_type, rep, estimator)
    if key in completed:
        print(f"[oms] skip {key}", flush=True)
        return completed

    bundle = ESTIMATOR_BUNDLES[estimator]
    print(
        f"[oms] start {scenario}/{dataset_id}/{shift_type} "
        f"rep={rep} est={estimator}",
        flush=True,
    )
    try:
        result = benchmark_permucate_drperm(
            df1,
            df2,
            feature_ind=feature_ind,
            seed=seed,
            n_perm=n_perm,
            **bundle,
        )
    except Exception as exc:
        print(f"[oms] FAIL {key}: {exc}", flush=True)
        return completed

    s_rows, m_rows = benchmark_rows_from_result(
        scenario,
        dataset_id,
        shift_type,
        result,
        rep=rep,
        estimator=estimator,
    )
    append_and_save(s_rows, m_rows, SCORES_CSV, METRICS_CSV)
    completed.add(key)
    for method in ("permucate", "drperm"):
        m = result["metrics"].get(method, {})
        print(
            f"[oms] done {method} est={estimator} "
            f"F1={m.get('F1', float('nan')):.3f} "
            f"AUC={m.get('selection_AUC', float('nan')):.3f}",
            flush=True,
        )
    gc.collect()
    return completed


def run_synthetic(estimators, n_rep, n_perm, completed):
    scenario = "synthetic"
    for si, (name, cfg) in enumerate(SYNTHETIC_SETTINGS.items()):
        for rep in range(n_rep):
            seed = 10_000 + si * 100 + rep
            X, Y, W = cfg["fn"](seed=seed, **cfg["kwargs"])
            df1, df2 = _xyw_to_batches(X, Y, W)
            feature_ind = np.asarray(CAUSAL_FEATURES, dtype=int)
            for est in estimators:
                completed = _run_cell(
                    scenario,
                    name,
                    cfg["shift_type"],
                    df1,
                    df2,
                    feature_ind,
                    est,
                    rep,
                    seed=seed + 17,
                    n_perm=n_perm,
                    completed=completed,
                )
    return completed


def _load_tabular(path, dataset_idx):
    if not os.path.isabs(path):
        path = os.path.join(TABULAR_DIR, os.path.basename(path))
    df = pd.read_csv(path)
    if "Unnamed: 0" in df.columns:
        df = df.drop(["Unnamed: 0"], axis=1)
    y_col = "Y" if "Y" in df.columns else df.columns[-1]
    feature_cols = [c for c in df.columns if c != y_col]
    X = df[feature_cols].apply(pd.to_numeric, errors="coerce")
    X = X.fillna(X.median(numeric_only=True)).fillna(0.0)
    Y = pd.to_numeric(df[y_col], errors="coerce").fillna(0.0)
    data = pd.concat([X, Y.rename("Y")], axis=1)
    part1, part2 = train_test_split(
        data, test_size=0.5, random_state=2000 + dataset_idx
    )
    return (
        np.asarray(part1.drop(columns=["Y"]), dtype=float),
        np.asarray(part1["Y"], dtype=float).ravel(),
        np.asarray(part2.drop(columns=["Y"]), dtype=float),
        np.asarray(part2["Y"], dtype=float).ravel(),
    )


def _subsample_xy(df1_X, df1_Y, df2_X, df2_Y, n_max, seed):
    rng = np.random.default_rng(seed)
    df1 = np.hstack([df1_X, df1_Y.reshape(-1, 1)])
    df2 = np.hstack([df2_X, df2_Y.reshape(-1, 1)])
    if df1.shape[0] > n_max:
        df1 = df1[rng.choice(df1.shape[0], n_max, replace=False)]
    if df2.shape[0] > n_max:
        df2 = df2[rng.choice(df2.shape[0], n_max, replace=False)]
    return df1[:, :-1], df1[:, -1].ravel(), df2[:, :-1], df2[:, -1].ravel()


def _apply_tabular_shift(shift_type, df1_X, df1_Y, df2_X, df2_Y, seed):
    if shift_type == "nonlinear_interaction":
        return synthetic_injection_nonlinear_interaction(
            df1_X, df1_Y, df2_X, df2_Y, n_prop=N_SHIFT_PROP, seed=seed
        )
    if shift_type == "linear":
        return synthetic_injection_linear(
            df1_X, df1_Y, df2_X, df2_Y, n_prop=N_SHIFT_PROP, seed=seed
        )
    if shift_type == "nonlinear_covariate_shift":
        return synthetic_covariate_shift_nonlinear_interaction(
            df1_X, df1_Y, df2_X, df2_Y, n_prop=N_SHIFT_PROP, seed=seed
        )
    raise ValueError(shift_type)



def run_tabular(estimators, n_rep, n_perm, completed):
    scenario = "tabular"
    for di, fname in enumerate(DATASET_LIST):
        dataset_id = os.path.splitext(fname)[0].replace("df_", "")
        try:
            base = _load_tabular(fname, di)
        except Exception as exc:
            print(f"[oms] tabular load fail {fname}: {exc}", flush=True)
            continue
        for shift_type in TABULAR_SHIFTS:
            for rep in range(n_rep):
                seed = 20_000 + di * 100 + rep
                df1_X, df1_Y, df2_X, df2_Y = _subsample_xy(
                    *base, SUBSAMPLE_N, seed
                )
                df1, df2, feature_ind = _apply_tabular_shift(
                    shift_type, df1_X, df1_Y, df2_X, df2_Y, seed + 3
                )
                feature_ind = np.asarray(feature_ind, dtype=int).ravel()
                for est in estimators:
                    completed = _run_cell(
                        scenario,
                        dataset_id,
                        shift_type,
                        df1,
                        df2,
                        feature_ind,
                        est,
                        rep,
                        seed=seed + 11,
                        n_perm=n_perm,
                        completed=completed,
                    )
    return completed


def _list_pair_files():
    out = {}
    for path in sorted(glob.glob(os.path.join("real_data", "*_pairdf.csv"))):
        m = PAIR_PATTERN.match(os.path.basename(path))
        if m:
            out[f"{m.group('s1')}_{m.group('s2')}"] = path
    return out


def _load_pairdf(path):
    df = pd.read_csv(path)
    if "Unnamed: 0" in df.columns:
        df = df.drop(["Unnamed: 0"], axis=1)
    y_col = "Y" if "Y" in df.columns else df.columns[-1]
    feature_cols = [c for c in df.columns if c != y_col]
    X = df[feature_cols].apply(pd.to_numeric, errors="coerce")
    X = X.fillna(X.median(numeric_only=True)).fillna(0.0)
    Y = pd.to_numeric(df[y_col], errors="coerce").fillna(0.0)
    arr = np.asarray(X, dtype=float)
    y = np.asarray(Y, dtype=float).ravel()
    half = len(y) // 2
    return arr[:half], y[:half], arr[half:], y[half:]


def _load_whyshift_pair(state1, state2, pair_path, seed, n_max):
    if pair_path and os.path.exists(pair_path):
        return _subsample_xy(*_load_pairdf(pair_path), n_max, seed)
    from whyshift import get_data

    df_X1, df_Y1, _ = get_data("income", state1, True, "./datasets/acs/", 2018)
    df_X2, df_Y2, _ = get_data("income", state2, True, "./datasets/acs/", 2018)
    return _subsample_xy(df_X1, df_Y1, df_X2, df_Y2, n_max, seed)


def _build_linear_whyshift(df1_X, df1_Y, df2_X, df2_Y, n_shift):
    beta1 = np.zeros(df1_X.shape[1])
    beta1[:n_shift] = np.linspace(1.5, 0.4, n_shift)
    df2_Y = df2_Y + df2_X @ beta1
    df1 = np.hstack([df1_X, df1_Y.reshape(-1, 1)])
    df2 = np.hstack([df2_X, df2_Y.reshape(-1, 1)])
    return df1, df2, np.arange(n_shift, dtype=int)





def run_whyshift(estimators, n_rep, n_perm, completed, n_pairs=8):
    scenario = "whyshift"
    pair_files = _list_pair_files()
    all_pairs = list(itertools.pairwise(WHYSHIFT_STATES))
    if pair_files:
        pairs = []
        for key in list(pair_files)[:n_pairs]:
            s1, s2 = key.split("_")
            pairs.append((s1, s2))
    else:
        # Prefer geographically spread pairs with ACS coverage.
        preferred = [
            ("NY", "FL"),
            ("IL", "OH"),
            ("WA", "OR"),
            ("MA", "PA"),
            ("GA", "NC"),
            ("CO", "AZ"),
            ("MI", "WI"),
            ("NJ", "VA"),
            ("MN", "IA"),
            ("TN", "AL"),
            ("OK", "KS"),
            ("NV", "UT"),
            ("CT", "RI"),
            ("SC", "LA"),
        ]
        pairs = preferred[:n_pairs]
    for pi, (state1, state2) in enumerate(pairs):
        dataset_id = f"{state1}_{state2}"
        for shift_type in ("linear", "nonlinear_covariate_shift"):
            for rep in range(n_rep):
                seed = 30_000 + pi * 20 + rep
                try:
                    df1_X, df1_Y, df2_X, df2_Y = _load_whyshift_pair(
                        state1,
                        state2,
                        pair_files.get(dataset_id),
                        seed=seed,
                        n_max=SUBSAMPLE_N,
                    )
                    if shift_type == "linear":
                        df1, df2, feature_ind = _build_linear_whyshift(
                            df1_X, df1_Y, df2_X, df2_Y, n_shift=12
                        )
                    else:
                        df1, df2, feature_ind = (
                            synthetic_covariate_shift_nonlinear_interaction(
                                df1_X,
                                df1_Y,
                                df2_X,
                                df2_Y,
                                n_prop=0.35,
                                seed=seed + 5,
                            )
                        )
                        feature_ind = np.asarray(feature_ind, dtype=int).ravel()
                except Exception as exc:
                    print(f"[oms] whyshift load fail {dataset_id}: {type(exc).__name__}: {exc!r}", flush=True)
                    continue
                for est in estimators:
                    completed = _run_cell(
                        scenario,
                        dataset_id,
                        shift_type,
                        df1,
                        df2,
                        feature_ind,
                        est,
                        rep,
                        seed=seed + 13,
                        n_perm=n_perm,
                        completed=completed,
                    )
    return completed


def _emb_path(year):
    return os.path.join(
        CHRONOBERG_EMB_DIR, f"year_{int(year)}_{CHRONOBERG_EMB_SUFFIX}"
    )


def _subsample_rows(X, n, seed):
    rng = np.random.default_rng(seed)
    X = np.asarray(X, dtype=float)
    if X.shape[0] <= n:
        return X
    return X[rng.choice(X.shape[0], size=n, replace=False)]


def _pca_pair(arr1, arr2, n_components=30, subsample=1500, seed=0):
    X1 = _subsample_rows(arr1, subsample, seed=seed)
    X2 = _subsample_rows(arr2, subsample, seed=seed + 1)
    n_comp = max(
        1,
        min(
            int(n_components),
            X1.shape[0] - 1,
            X1.shape[1],
            X2.shape[0] - 1,
            X2.shape[1],
        ),
    )
    pca = PCA(n_components=n_comp, random_state=2000)
    X1_p = pca.fit_transform(X1)
    X2_p = pca.transform(X2)
    return X1_p, X2_p


def run_chronoberg(estimators, n_rep, n_perm, completed):
    scenario = "chronoberg"
    for yi, start_year in enumerate(CHRONOBERG_YEARS):
        end_year = int(start_year) + 10
        dataset_id = f"{start_year}_{end_year}"
        path1, path2 = _emb_path(start_year), _emb_path(end_year)
        if not os.path.exists(path1) or not os.path.exists(path2):
            # Fall back to nearest available year files.
            avail = sorted(
                int(n.split("_")[1])
                for n in os.listdir(CHRONOBERG_EMB_DIR)
                if n.startswith("year_") and n.endswith(CHRONOBERG_EMB_SUFFIX)
            )
            if len(avail) < 2:
                print("[oms] chronoberg: no embeddings", flush=True)
                return completed
            # Pick evenly spaced available pairs.
            idx = np.round(np.linspace(0, len(avail) - 2, len(CHRONOBERG_YEARS))).astype(int)
            start_year = avail[idx[yi]]
            end_year = avail[min(idx[yi] + 1, len(avail) - 1)]
            if end_year <= start_year:
                end_year = avail[min(idx[yi] + 5, len(avail) - 1)]
            dataset_id = f"{start_year}_{end_year}"
            path1, path2 = _emb_path(start_year), _emb_path(end_year)
            if not os.path.exists(path1) or not os.path.exists(path2):
                continue
        X1, X2 = _pca_pair(np.load(path1), np.load(path2), seed=4000 + yi)
        y1 = np.arange(X1.shape[0], dtype=float)
        y2 = np.arange(X2.shape[0], dtype=float)
        for adv in CHRONOBERG_ADV:
            for rep in range(n_rep):
                seed = 40_000 + yi * 10 + rep
                try:
                    X_adv, Y_t, feature_ind, n1 = impose_adv_shift(
                        X1, y1, X2, y2, method=adv, task="reg"
                    )
                    df1 = np.hstack([X_adv[:n1], Y_t[:n1].reshape(-1, 1)])
                    df2 = np.hstack([X_adv[n1:], Y_t[n1:].reshape(-1, 1)])
                    feature_ind = np.asarray(feature_ind, dtype=int).ravel()
                except Exception as exc:
                    print(f"[oms] chronoberg fail {dataset_id}/{adv}: {exc}", flush=True)
                    continue
                for est in estimators:
                    completed = run_individual_version(
                        scenario,
                        dataset_id,
                        adv,
                        df1,
                        df2,
                        feature_ind,
                        est,
                        rep,
                        seed=seed + 7,
                        n_perm=n_perm,
                        completed=completed,
                    )
    return completed