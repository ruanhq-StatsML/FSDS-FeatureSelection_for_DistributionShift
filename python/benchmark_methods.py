#benchmark methods:
import os
os.chdir(
    "/Users/heqiaoruan/Library/Mobile Documents/com~apple~CloudDocs/Documents/GitHub 2/Causal_Objective_Permutation_Test/Python"
)
import json
import time
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
import numpy as np
import pandas as pd
import shap
try:
    import SGShift
except ImportError:
    SGShift = None
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from VIMP_mmd_benchmark import MMD, _rank_positions
from grf_vimp_causalForest import cf_variable_importance
from LOCO_vimp_r_risk import vimp_loco_r_risk
from permuCATE_vimp import permuCATE_vimp

MODEL_M = "rf_regressor"
MODEL_E = "logistic_classifier"
MODEL_TAU = "rf_regressor"
MODEL_NU = "rf_regressor"
N_PERM_CATE = 20
N_ESTIMATORS = 120

def metrics_from_scores(scores, feature_ind, p):
    k = len(feature_ind)
    selected = np.argsort(-np.asarray(scores, dtype=float))[:k]
    return metrics(selected, feature_ind, p, scores=scores)

def metrics(selected, shift_index, p, scores=None):
    sel = set(int(i) for i in selected)
    true = set(int(i) for i in shift_index)
    tp = len(sel & true)
    fp = len(sel - true)
    fn = len(true - sel)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0.0 else 0.0
    fdr = fp / (tp + fp) if (tp + fp) > 0 else 0.0
    out = {
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "F1": round(f1, 4),
        "FDR": round(fdr, 4),
    }
    if scores is not None:
        membership = np.zeros(p, dtype=int)
        membership[list(true)] = 1
        sc = np.nan_to_num(np.asarray(scores, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
        if 0 < membership.sum() < p:
            out["selection_AUC"] = round(float(roc_auc_score(membership, sc)), 4)
    return out

def _split_batches(df1, df2):
    p = df1.shape[1] - 1
    df1_X = np.asarray(df1[:, :p], dtype=float)
    df2_X = np.asarray(df2[:, :p], dtype=float)
    df1_Y = np.asarray(df1[:, p], dtype=float).ravel()
    df2_Y = np.asarray(df2[:, p], dtype=float).ravel()
    return df1_X, df1_Y, df2_X, df2_Y, p



'''
Compute the delta shapley value here.
'''
def compute_delta_shap(
    df1_X,
    df1_Y,
    df2_X,
    df2_Y,
    test_size=0.3,
    seed=2047,
):
    rng = np.random.default_rng(seed)
    p = df1_X.shape[1]
    n1 = df1_X.shape[0]
    n2 = df2_X.shape[0]
    rf_kwargs = dict(
        max_depth=max(2, round(p // 3)),
        n_estimators=100,
        min_samples_leaf=max(1, round(np.sqrt(n1 + n2) // 2)),
        n_jobs=1,
        random_state=int(rng.integers(0, 10000)),
    )
    model1 = RandomForestRegressor(**rf_kwargs)
    model2 = RandomForestRegressor(**{**rf_kwargs, "random_state": int(rng.integers(0, 10000))})
    X_train1, X_test1, Y_train1, _ = train_test_split(
        df1_X, df1_Y, test_size=test_size, random_state=seed + 10
    )
    X_train2, X_test2, Y_train2, _ = train_test_split(
        df2_X, df2_Y, test_size=test_size, random_state=seed + 10
    )
    model1.fit(X_train1, Y_train1)
    model2.fit(X_train2, Y_train2)
    explainer1 = shap.Explainer(model1, X_train1)
    shap_value_b1 = explainer1(X_test1, check_additivity=False)
    explainer2 = shap.Explainer(model2, X_train2)
    shap_value_b2 = explainer2(X_test2, check_additivity=False)
    mean_shap_b1 = np.mean(shap_value_b1.values, axis=0)
    mean_shap_b2 = np.mean(shap_value_b2.values, axis=0)
    delta_shap = np.abs(mean_shap_b2 - mean_shap_b1)
    return delta_shap, _rank_positions(delta_shap)


#LOCO + MMD: Leave-one-covariate-out and Re-calculate the MMD.
def compute_loco_mmd_batches(df1_X, df2_X, max_n=1000, seed=0):
    rng = np.random.default_rng(seed)
    X_exist = _subsample_batch(np.asarray(df1_X, dtype=float), max_n, rng)
    X_new = _subsample_batch(np.asarray(df2_X, dtype=float), max_n, rng)
    mmd_test = MMD(compute_kernel="rbf")
    mmd_orig, _ = mmd_test(X_exist, X_new)
    p = X_exist.shape[1]
    mmd_dic = np.zeros(p, dtype=float)
    for j in range(p):
        mmd_perm, _ = mmd_test(np.delete(X_exist, j, axis=1), np.delete(X_new, j, axis=1))
        mmd_dic[j] = mmd_perm
    mmd_vimp = mmd_orig - mmd_dic
    return mmd_vimp, _rank_positions(mmd_vimp)

#Random Forest as a Domain Classifier:
def compute_rf_domain_vimp(df1_X, df2_X, seed=0):
    X = np.vstack([df1_X, df2_X])
    W = np.concatenate([np.zeros(len(df1_X)), np.ones(len(df2_X))]).astype(int)
    p = X.shape[1]
    clf = RandomForestClassifier(
        n_estimators=100,
        max_depth=max(2, int(np.round(np.sqrt(p)))),
        min_samples_leaf=max(1, int(np.round(np.sqrt(len(X)) // 2))),
        n_jobs=1,
        random_state=seed,
    )
    clf.fit(X, W)
    return clf.feature_importances_, _rank_positions(clf.feature_importances_)



'''
The benchmark from the existing works:
https://arxiv.org/abs/2505.20634
- linear
- nonlinear
- knockoff based procedure

'''
def sgs_shift_benchmark(
    df1_X,
    df1_Y,
    df2_X,
    df2_Y,
    task="regression",
    model_type="random_forest_reg",
    selection_tol=1e-4,
    random_state=24,
    lambdas=None,
    v=3,
    B=20,
    pi_thr=0.5,
):
    if SGShift is None:
        raise ImportError("SGShift module unavailable")
    if lambdas is None:
        lambdas = list(np.logspace(-2.5, 0.3, 8))
    solver = SGShift.fit_model(task, model_type, random_state, df1_X, df1_Y)
    beta_mis, delta_mis, _ = SGShift.cross_validate_lambda(
        df2_X, df2_Y, solver, lambdas, misspec=True, X_S=df1_X, y_S=df1_Y, task=task
    )
    delta_unmis, _ = SGShift.cross_validate_lambda(
        df2_X, df2_Y, solver, lambdas, misspec=False, X_S=df1_X, y_S=df1_Y, task=task
    )
    scores_l1mis = np.abs(delta_mis)
    scores_l1unmis = np.abs(delta_unmis)
    selected_l1mis = np.where(scores_l1mis > selection_tol)[0]
    selected_l1unmis = np.where(scores_l1unmis > selection_tol)[0]
    pi = SGShift.derandom_knock(
        solver, df1_X, df1_Y, df2_X, df2_Y, B, lambdas, v=v, task=task
    )
    freq = np.max(np.vstack([pi[l] for l in lambdas]), axis=0)
    selected_knock = np.where(freq >= pi_thr)[0]
    return {
        "l1_score_misspec": scores_l1mis,
        "l1_score_spec": scores_l1unmis,
        "knock_score": freq,
        "l1_selected_misspec": selected_l1mis,
        "l1_selected_spec": selected_l1unmis,
        "selected_knock": selected_knock,
    }



'''
The three types of the variable importance from the causal inference methods.
'''
def run_all_causal_vimp(X, Y, W, seed=0):
    perm_vimp = permuCATE_vimp(
        X,
        Y,
        W,
        model_m=MODEL_M,
        model_e=MODEL_E,
        model_tau=MODEL_TAU,
        model_nu=MODEL_NU,
        n_perm=N_PERM_CATE,
        seed=seed,
    )
    loco_vimp = vimp_loco_r_risk(X, Y, W, model_m=MODEL_M, model_e=MODEL_E, seed=seed)
    cf_vimp = cf_variable_importance(
        X, Y, W, model_psm="logistic", model_outcome="rf", n_estimators=N_ESTIMATORS, seed=seed
    )
    return {
        "permucate": perm_vimp,
        "loco_r_risk": loco_vimp,
        "grf": cf_vimp,
    }


'''
The benchmark evaluation procedures - the methods.
'''
def benchmark_whole_feature_selection(
    df1,
    df2,
    feature_ind,
    seed=0,
    max_mmd_n=1000,
):
    time_dict = {}
    feature_ind = np.asarray(feature_ind, dtype=int).ravel()
    df1_X, df1_Y, df2_X, df2_Y, p = _split_batches(df1, df2)
    df1_X = np.nan_to_num(np.asarray(df1_X, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    df2_X = np.nan_to_num(np.asarray(df2_X, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    n1, n2 = len(df1_X), len(df2_X)
    scores = {}
    metrics_result = {}
    t0 = time.perf_counter()
    rf_scores, _ = compute_rf_domain_vimp(df1_X, df2_X, seed=seed)
    time_dict["rf"] = float(time.perf_counter() - t0)
    scores["rf_domain"] = rf_scores
    metrics_result["rf_domain"] = metrics_from_scores(rf_scores, feature_ind, p)
    t0 = time.perf_counter()
    delta_shap, _ = compute_delta_shap(df1_X, df1_Y, df2_X, df2_Y, seed=seed)
    time_dict["shap"] = float(time.perf_counter() - t0)
    scores["delta_shap"] = delta_shap
    metrics_result["delta_shap"] = metrics_from_scores(delta_shap, feature_ind, p)
    t0 = time.perf_counter()
    mmd_vimp, _ = compute_loco_mmd_batches(df1_X, df2_X, max_n=max_mmd_n, seed=seed)
    time_dict["mmd"] = float(time.perf_counter() - t0)
    scores["loco_mmd"] = mmd_vimp
    metrics_result["loco_mmd"] = metrics_from_scores(mmd_vimp, feature_ind, p)
    X = np.vstack([df1_X, df2_X])
    Y = np.concatenate([df1_Y, df2_Y])
    W = np.concatenate([np.zeros(n1), np.ones(n2)])
    t0 = time.perf_counter()
    causal = run_all_causal_vimp(X, Y, W, seed=seed + 4)
    time_dict["permucate"] = float(time.perf_counter() - t0) / 3.0
    time_dict["loco"] = time_dict["permucate"]
    time_dict["grf"] = time_dict["permucate"]
    for name, vimp in causal.items():
        scores[name] = vimp
        metrics_result[name] = metrics_from_scores(vimp, feature_ind, p)
    try:
        from DRPerm import compute_drperm_vimp
        t0 = time.perf_counter()
        drperm_vimp = compute_drperm_vimp(X, Y, W, seed=seed + 6)
        time_dict["drperm"] = float(time.perf_counter() - t0)
        scores["drperm"] = drperm_vimp
        metrics_result["drperm"] = metrics_from_scores(drperm_vimp, feature_ind, p)
    except Exception as exc:
        print(f"[realdata] DRPerm skipped: {exc}", flush=True)
    try:
        t0 = time.perf_counter()
        sgs = sgs_shift_benchmark(df1_X, df1_Y, df2_X, df2_Y, random_state=seed + 8)
        sgs_time = float(time.perf_counter() - t0)
        scores["l1_misspec"] = sgs["l1_score_misspec"]
        scores["l1_spec"] = sgs["l1_score_spec"]
        scores["knockoff"] = sgs["knock_score"]
        time_dict["l1_misspec"] = sgs_time / 3.0
        time_dict["l1spec"] = sgs_time / 3.0
        time_dict["knockoff"] = sgs_time / 3.0
        metrics_result["l1_misspec"] = metrics_from_scores(sgs["l1_score_misspec"], feature_ind, p)
        metrics_result["l1_spec"] = metrics_from_scores(sgs["l1_score_spec"], feature_ind, p)
        metrics_result["knockoff"] = metrics_from_scores(sgs["knock_score"], feature_ind, p)
    except Exception as exc:
        print(f"[realdata] SGShift skipped: {exc}", flush=True)
    return {
        "time_profile": time_dict,
        "scores": scores,
        "metrics": metrics_result,
        "feature_ind": feature_ind.tolist(),
        "p": int(p),
    }




