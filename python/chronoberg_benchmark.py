
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
    metrics_from_scores,
    sgs_shift_benchmark
)
from real_data_utils import *
from adversarial_perturbation_distribution_shift_whole import impose_adv_shift


EMB_DIR = 'real_data/yearly_extracted_batch'
EMB_NAME = 'sentence_chronoberg_processed_Emb_gemma_embedding2.npy'
START_YEAR = np.round(np.linspace(1750, 1970, 23)).astype(int)

METHOD_ADV = [
    "miFGSM",
    "sini_FGSM",
    "vmi_FGSM",
    "rFGSM",
    "jitter",
]

SCENARIO = "chronoberg"
SUBSAMPLE = 2000
PCA_N = 30
RANDOM_STATE = 2000



#with 10 years in between:
for pair_idx, start_year in enumerate(START_YEAR):
    end_year = int(start_year) + 10
    dataset_id = f"{start_year}_{end_year}"
    #start year path:
    path1 = os.path.join(EMB_DIR, f"year_{int(start_year)}_{EMB_NAME}")
    path2 = os.path.join(EMB_DIR, f"year_{int(end_year)}_{EMB_NAME}")
    df1_arr = np.load(path1)
    df2_arr = np.load(path2)
    df1_X = arr1[:, :-1]
    df2_X = arr2[:, :-1]
    df1_Y = arr1[:, -1].ravel()
    df2_Y = arr2[:, -1].ravel()
    for i, method in enumerate(METHOD_ADV):
    	dataset_id = f"{start_year}_{end_year}"
    	#Initiate the keys here:
        key = ('chronoberg', dataset_id, method, 0)
        X_adv_t, Y_t, feature_ind, n1 = impose_adv_shift(
            df1_X, df1_Y, df2_X, df2_Y, method=method, task="reg"
        )
        df1 = np.hstack([X_adv_t[:n1], Y_t[:n1].reshape(-1, 1)])
        df2 = np.hstack([X_adv_t[n1:], Y_t[n1:].reshape(-1, 1)])
        results = benchmark_whole_feature_selection(
            df1, df2, feature_ind, seed=2000 + pair_idx * 10 + i
        )
        row_scores_df, row_metrics_df = save_result(
            'chronoberg', dataset_id, method, results,
            scores_csv = f'chronoberg_benchmark_scores{start_year}_{end_year}.csv',
            metrics_csv = f'chronoberg_benchmark_metrics{start_year}_{end_year}.csv',
            rep=0
        )
        save_json_result(f'chronoberg_benchmark_results{start_year}_{end_year}.json',
            f"'chronoberg'::{dataset_id}::{method}", results)





















