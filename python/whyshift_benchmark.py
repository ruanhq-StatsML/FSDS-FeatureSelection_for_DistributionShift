#utils for real-data applications:
import os
import gc
import json
import itertools
import numpy as np
from sklearn.preprocessing import LabelEncoder
from whyshift import get_data
from real_data_utils import (
  synthetic_injection_linear,
  pca_pair, preprocess_whyshift,
  build_pair_whyshift,
  impose_linear_cd_whyshift,
  save_result
)

os.chdir(
    "/Users/heqiaoruan/Library/Mobile Documents/com~apple~CloudDocs/Documents/GitHub 2/Causal_Objective_Permutation_Test/Python"
)


LIST_STATES = [
    "AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "FL",
    "GA", "HI", "ID", "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD",
    "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ", "NM",
    "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI", "SC", "SD", "TN",
    "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY", "PR",
]
SUBSAMPLE = 2500
OUT_DIR = "real_data"
RNG = np.random.default_rng(2018)



def load_tabular_dataset(path, dataset_idx):
    df = pd.read_csv(path)
    if "Unnamed: 0" in df.columns:
        df = df.drop(['Unnamed: 0'], axis = 1)
    df_part1, df_part2 = train_test_split(
        df, test_size = 0.5, random_state = 2000 + dataset_idx
    )
    df1_X = np.asarray(df_part1.drop(columns = ['Y'], error = 'ignore'))
    if "Y" in df_part1.columns:
        df1_Y = np.asarray(df_part1['Y'], dtype = float).ravel()
        df2_Y = np.asarray(df_part2['Y'], dtype = float).ravel()
        df2_X = np.asarray(df_part2.drop(['Y'], axis = 1))
    else:
        df1_Y = np.asarray(df_part1.iloc[:, -1], dtype = float).ravel()
        df2_Y = np.asarray(df_part2.iloc[:, -1], dtype = float).ravel()
        df2_X = np.asarray(df_part2.iloc[:, :-1])
    return df1_X, df1_Y, df2_X, df2_Y


'''
For each of the pair dataset, we impose the shift on the first 11 features.
'''
seed_counter = 10
for state1, state2 in list(itertools.pairwise(LIST_STATES)):
    dataset_id = f"{state1}_{state2}"
    key  = ("whyshift", dataset_id, "linear", 0)
    if key in completed:
        print(f"[Whyshift linear] skip {dataset_id}", flush = True)
        continue
    df_X1, df_Y1, _ = get_data('income', state1, True, './dataset/acs/', 2018)
    df_X2, df_Y2, _ = get_data('income', state2, True, './dataset/acs/', 2018)
    beta1 = np.zeros(df_X1.shape[1])
    beta1[:11] = np.linspace(1.5, 0.5, 11)
    df_Y2 = df_Y2 + df_X2 @ beta1
    df_1 = np.hstack([df_X1, df_Y1.reshape(-1, 1)])
    df_2 = np.hstack([df_X2, df_Y2.reshape(-1, 1)])
    seed_counter += 1
    rng1 = np.random.default_rng(seed_counter)
    seed_counter += 2
    rng2 = np.random.default_rng(seed_counter)
    idx1 = rng1.integers(0, df_1.shape[0], subsample_size)
    idx2 = rng2.integers(0, df_2.shape[0], subsample_size)
    df1 = df_1[idx1, :]
    df2 = df_2[idx2, :]
    feature_ind = np.arange(11)
    #Benchmark feature selection:
    result = benchmark_whole_feature_selection(
        df1, df2, feature_ind = feature_ind,
        seed = 500 + seed_counter
    )
    #appending the results into the dataset:
    s_rows, m_rows = benchmark_rows_from_result(
        "whyshift", dataset_id, 'linear', 
        result, rep = 0
    )
    rows_scores.extend(s_rows)
    rows_metrics.extend(m_rows)
    save_result(result, 
        "whyshift", dataset_id, 'linear',
        result)
    save_json_result(f'whyshift_linear_{dataset_id}_benchmark.json', result)
    rows_scores.clear()
    rows_metrics.clear()
    gc.collect()
    print(f"Whyshift {dataset_id} finished")


















 





























































