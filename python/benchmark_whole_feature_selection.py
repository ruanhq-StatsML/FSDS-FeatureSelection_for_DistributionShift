import os
os.chdir(
    "/Users/heqiaoruan/Documents/GitHub 2/Causal_Objective_Permutation_Test/Python/FSDS_Software"
)
import shap
import torch
import SGShift
import numpy as np
import pandas as pd
from datafix import DFLocate
from fslnet.fslnet import FSLNet
from sklearn.datasets import fetch_openml
from sklearn.metrics import roc_auc_score
from permuCATE_vimp import permuCATE_vimp
from VIMP_drperm_benchmark import DRPerm_LOCO
from LOCO_vimp_r_risk import vimp_loco_r_risk
from sklearn.model_selection import train_test_split
from grf_vimp_causalForest import cf_variable_importance
from benchmark_localization_net import benchmark_localization_net
from domain_classifier_VIMP_whole import compute_vimp_from_adapter
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from other_benchmark_method import (
    compute_loco_mmd_batches,
    compute_rf_domain_vimp,
    rf_domain_classifier,
    compute_delta_shap,
    pca_covariate_shift_pair,
    run_all_causal_vimp,
    compute_vimp_from_adapter
)

'''
Benchmark all the methods for feature selection of distribution shift:
Input: 
df1:             Existing Batch of Data.
df2:             New Batch of Data.
feature_ind:     The indices of the shifted features.
test_size:       The proportion of the test size in the rf-domain classifier like procedure.
seed:            The random seed.
max_mmd_n:       The maximum sample-size for the MMD procedure.
max_hsic_n:      The maximum sample-size for the HSIC procedure. 
model_m:         The outcome model, by default 'rf_regressor'.
model_e:         The propensity score model, by default 'rf_classifier'.
model_tau:       The pseudo-outcome regression model, by default 'rf_regressor'.
model_nu:        The conditional dependence model, by defautl 'rf_regressor'.
n_permu_cate:    The number of permutation in the PermuCATE procedure.
rf_domain_type:  The list of the methods in the class of the domain classifier, 
                 including "shadowVIMP", "Boruta", "vitaPIMP", "condperm", "LOCO".
n_estimators:    The number of estimators/causal trees in the feature selection procedure.

Return:
result_dict:     The result dictionary for the benchmark feature selection results, including the metrics for each of the method in this procedure.
'''
RFDOMAIN_TYPE = ['shadowVIMP', 'Boruta', 'vitaPIMP', 'condperm', 'LOCO']
MODEL_M = 'rf_regressor'
MODEL_E = 'rf_classifier'
MODEL_PO = 'rf_regressor'
MODEL_TAU = 'rf_regressor'
MODEL_NU = 'rf_regressor'
N_PERMU_CATE = 25
N_ESTIMATORS = 120
def benchmark_whole_feature_selection(
    df1,
    df2,
    feature_ind,
    test_size = 0.3,
    seed=0,
    max_mmd_n=1000,
    max_hsic_n=1000,
    hsic_sigma=1.0,
    model_m = MODEL_M,
    model_e = MODEL_E,
    model_tau = MODEL_TAU,
    model_nu = MODEL_NU,
    n_permu_cate = N_PERMU_CATE,
    rf_domain_type = RF_DOMAIN_TYPE,
    n_estimators = N_ESTIMATORS
):
    #Initiate the summary statistics:
    time_dict = {}
    scores = {}
    metrics_result = {}
    feature_ind = np.asarray(feature_ind, dtype=int).ravel()
    df1_X, df1_Y, df2_X, df2_Y, p = split_subset(df1, df2)
    df1_X = np.nan_to_num(np.asarray(df1_X, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    df2_X = np.nan_to_num(np.asarray(df2_X, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    n1, n2 = len(df1_X), len(df2_X)
    t0 = time.perf_counter()
    rf_scores, _ = compute_rf_domain_vimp(df1_X, df2_X, seed=seed)
    time_dict['rf_domain'] = float(time.perf_counter() - t0)
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
    t0 = time.perf_counter()
    hsic_vimp, _ = compute_hsic_loco_batches(
        df1_X,
        df2_X,
        max_n=max_hsic_n,
        seed=seed + 1,
        sigma=hsic_sigma,
    )
    time_dict["hsic"] = float(time.perf_counter() - t0)
    scores["loco_hsic"] = hsic_vimp
    metrics_result["loco_hsic"] = metrics_from_scores(hsic_vimp, feature_ind, p)
    X = np.vstack([df1_X, df2_X])
    Y = np.concatenate([df1_Y, df2_Y])
    W = np.concatenate([np.zeros(n1), np.ones(n2)])
    t0 = time.perf_counter()
    perm_vimp = permuCATE_vimp(
        X,
        Y,
        W,
        model_m = model_m,
        model_e = model_e,
        model_tau = model_tau,
        model_nu = model_nu,
        n_perm = n_permu_cate,
        seed = seed + 4,
    )
    time_dict["permucate"] = float(time.perf_counter() - t0)
    scores["permucate"] = perm_vimp
    metrics_result["permucate"] = metrics_from_scores(perm_vimp, feature_ind, p)
    t0 = time.perf_counter()
    loco_vimp = vimp_loco_r_risk(X, Y, W, model_m=model_m, model_e=model_e, seed=seed + 4)
    time_dict["loco_r_risk"] = float(time.perf_counter() - t0)
    scores["loco_r_risk"] = loco_vimp
    metrics_result["loco_r_risk"] = metrics_from_scores(loco_vimp, feature_ind, p)
    t0 = time.perf_counter()
    cf_vimp = cf_variable_importance(
        X, Y, W, model_psm="logistic", model_outcome="rf", n_estimators=n_estimators, seed=seed + 4
    )
    time_dict["grf"] = float(time.perf_counter() - t0)
    scores["grf"] = cf_vimp
    metrics_result["grf"] = metrics_from_scores(cf_vimp, feature_ind, p)
    t0 = time.perf_counter()
    drperm_out = DRPerm_LOCO(X, Y, W, seed=seed + 6)
    time_dict["drperm"] = float(time.perf_counter() - t0)
    drperm_vimp = drperm_out["po_risk_vimp"]
    scores["drperm"] = drperm_vimp
    metrics_result["drperm"] = metrics_from_scores(drperm_vimp, feature_ind, p)
    t0 = time.perf_counter()
    sgs = sgs_shift_benchmark(df1_X, df1_Y, df2_X, df2_Y, random_state=seed + 8)
    sgs_time = float(time.perf_counter() - t0)
    scores["l1_misspec"] = sgs["l1_score_misspec"]
    scores["l1_spec"] = sgs["l1_score_spec"]
    scores["knockoff"] = sgs["knock_score"]
    time_dict["l1_misspec"] = float(sgs_time / 3.0)
    time_dict["l1spec"] = float(sgs_time / 3.0)
    time_dict["knockoff"] = float(sgs_time / 3.0)
    metrics_result["l1_misspec"] = metrics_from_scores(sgs["l1_score_misspec"], feature_ind, p)
    metrics_result["l1_spec"] = metrics_from_scores(sgs["l1_score_spec"], feature_ind, p)
    metrics_result["knockoff"] = metrics_from_scores(sgs["knock_score"], feature_ind, p)
    #Benchmark the Datafix and the FSLNet:
    t0 = time.perf_counter()
    result_localization = benchmark_localization_net(df1, df2, feature_ind,
            threshold = 0.7, cv_fold = 5, B = 15)
    metric_results['fslnet'] = result_localization['fslnet']
    metric_results['datafix'] = result_localization['datafix']
    localization_net_time = float(time.perf_counter() - t0)
    time_dict['fslnet'] = float(localization_net_time / 2.0)
    time_dict['datafix'] = float(localization_net_time / 2.0)
    #Benchmark the other adapter based methodologies 
    adapter_RF = RandomForestClassifier(
        max_features = np.sqrt(p)/p,
        n_estimators = 150,
        max_depth = 5,
        min_samples_leaf = max(1, round(np.sqrt(p)//2)),
        n_jobs = 1,
        random_state = seed + 1
    )
    fitted_obj = model_rf['fit'](X_train, Y_train)
    for vimp_type in rf_domain_type:
        t0 = perf.time_counter()
        vimp_value, vimp_rank = compute_vimp_from_adapter(adapter_RF,
            fitted_obj, X_train, Y_train, X_test, Y_test, vimp_type = vimp_type)
        time_dict[str(vimp_type) + 'RFDomain'] = float(time.perf_counter() - t0)
        metric_results[str(vimp_type) + 'RFDomain'] = metrics_from_scores(vimp_value, feature_ind, p)
    return {
        "time_profile": time_dict,
        "scores": scores,
        "metrics": metrics_result,
        "feature_ind": feature_ind.tolist(),
        "p": int(p),
    }


'''


'''
