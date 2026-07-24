#FMoW dataset:
import os
os.chdir(
    "/Users/heqiaoruan/Library/Mobile Documents/com~apple~CloudDocs/Documents/GitHub 2/Causal_Objective_Permutation_Test/Python"
)
import numpy as np
import pandas as pd
from itertools import product
from sklearn.decomposition import PCA
from benchmark_methods import (
    benchmark_whole_feature_selection,
    save_result,
    save_json_result
)
from adversarial_perturbation_distribution_shift_whole import impose_adv_shift
pairwise_list = [
    "space_facility",
    "airport",
    "debris_or_rubble",
    "border_checkpoint",
    "port",
]
METHOD_ADV = [
    "miFGSM",
    "sini_FGSM",
    "vmi_FGSM",
    "rFGSM",
    "jitter",
]
RANDOM_STATE = 2000
SUBSAMPLE = 2000
PCA_NUMBER = 30



#Impose various types of the benchmarks:
for grp1, grp2 in product(pairwise_list, pairwise_list):
    if grp1 == grp2:
        continue
    dataset_id = f"{grp1}_{grp2}"
    df1 = pd.read_csv(f"real_data/df_{grp1}.csv")
    df2 = pd.read_csv(f"real_data/df_{grp2}.csv")
    arr1, arr2 = pca_pair(raw1, raw2)
    df1_X = arr1[:, :-1]
    df2_X = arr2[:, :-1]
    df1_Y = arr1[:, -1].ravel()
    df2_Y = arr2[:, -1].ravel()
    for i, method in enumerate(METHOD_ADV):
        key = (SCENARIO, dataset_id, method, 0)
        #Impose the certain type of shift here:
        X_adv_t, Y_t, feature_ind, n1 = impose_adv_shift(
            df1_X, df1_Y, df2_X, df2_Y, method=method, task="reg"
        )
        df1 = np.hstack([X_adv_t[:n1], Y_t[:n1].reshape(-1, 1)])
        df2 = np.hstack([X_adv_t[n1:], Y_t[n1:].reshape(-1, 1)])
        results = benchmark_whole_feature_selection(
            df1, df2, feature_ind, seed=2000 + i
        )
        row_scores_df, row_metrics_df = save_result(
            SCENARIO, dataset_id, method, results,
            scores_csv = 'FMoW_benchmark_scores.csv',
            metrics_csv = 'FMoW_benchmark_metrics.csv',
            rep=0
        )
        save_result(s_rows, m_rows, 'FMoW_benchmark_scores.csv',
            'FMoW_benchmark_metrics.csv')
        save_json_result('FMoW_benchmark_results.json',
            f"{SCENARIO}::{dataset_id}::{method}", results)





























