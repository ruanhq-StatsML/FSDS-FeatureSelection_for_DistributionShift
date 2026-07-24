#utils for real-data applications:
import numpy as np
import gc
import itertools
import json
import os

os.chdir(
    "/Users/heqiaoruan/Library/Mobile Documents/com~apple~CloudDocs/Documents/GitHub 2/Causal_Objective_Permutation_Test/Python"
)

INPUT_PATH = 'real_data'
OUTPUT_PATH = 'real_data'

def synthetic_injection_linear(df1_X, df1_Y, df2_X, df2_Y, n_prop=0.3, seed=0):
    rng = np.random.default_rng(seed)
    n2, p = df2_X.shape
    n_shift = max(1, int(np.round(p * n_prop)))
    feature_ind = rng.choice(p, size=n_shift, replace=False)
    beta_shift = np.zeros(p)
    beta_shift[feature_ind] = np.linspace(0, 1, len(feature_ind))
    df2_Y = df2_Y + df2_X @ beta_shift + rng.normal(0, 0.05, n2)
    df1 = np.hstack([df1_X, np.asarray(df1_Y, dtype=float).reshape(-1, 1)])
    df2 = np.hstack([df2_X, np.asarray(df2_Y, dtype=float).reshape(-1, 1)])
    return df1, df2, feature_ind

'''
Paired PCA dimensionality reduction here:
'''
def pca_pair(df1, df2, target_pca=PCA_TARGET, random_state=RANDOM_STATE, subsample=SUBSAMPLE, seed=0):
    if "Unnamed: 0" in df1.columns:
        df1 = df1.drop(["Unnamed: 0"], axis=1)
    if "Unnamed: 0" in df2.columns:
        df2 = df2.drop(["Unnamed: 0"], axis=1)
    X1 = df1.apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=float)
    X2 = df2.apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=float)
    rng = np.random.default_rng(seed)
    if X1.shape[0] > subsample:
        X1 = X1[rng.choice(X1.shape[0], subsample, replace=False)]
    if X2.shape[0] > subsample:
        X2 = X2[rng.choice(X2.shape[0], subsample, replace=False)]
    n_comp = adaptive_pca_n(X1.shape[0], X2.shape[0], X1.shape[1], target_pca)
    pca = PCA(n_components=n_comp, random_state=random_state)
    X1_p = pca.fit_transform(X1)
    X2_p = pca.transform(X2)
    n1, n2 = X1_p.shape[0], X2_p.shape[0]
    out1 = np.hstack([X1_p, np.arange(n1, dtype=float).reshape(-1, 1)])
    out2 = np.hstack([X2_p, (np.arange(n2, dtype=float) - n2).reshape(-1, 1)])
    return out1, out2

'''
Process the whyshift dataset via the following procedures:
[1] If the number of unique levels in a certain feature is less or equal to 5,
    then the label-encoder is imposed.
[2] Remove the features with only 1 level which has no predictive power in the responses Y.
'''
def preprocess_whyshift(X1, X2, feature_names):
    n_unique_levels = np.apply_along_axis(lambda X: len(np.unique(X)),
        axis = 0, arr = X1)
    drop_col = [j for j in np.arange(len(n_unique_levels)) if n_unique_levels[j] == 1]
    for k in range(len(n_unique_levels)):
        #Two types -> making it the label encoder if the types of unique values is less than 5
        if n_unique_levels[k] <= 5:
            enc = LabelEncoder()
            X1[:, k] = enc.fit_transform(X1[:, k])
            enc = LabelEncoder()            
            X2[:, k] = enc.fit_transform(X2[:, k])
    maintained_col = np.setdiff(np.arange(X1.shape[1]), drop_col)
    X1 = X1[:, maintained_col]
    X2 = X2[:, maintained_col]
    maintained_feature_names = np.array(feature_names)[maintained_col]
    return X1, X2, maintained_feature_names

 
'''
Building the pairs of data in the whyshift datasets across different states:
https://github.com/namkoong-lab/whyshift - The dataset is leveraged from the whyshift.
'''
OUT_DIR = 'real_data'
def build_pair_whyshift(state1, state2, subsample_size = SUBSAMPLE_SIZE)
    os.makedirs(OUT_DIR, exist_ok = True)
    out_path = f"{OUT_DIR}/{state1}_{state2}_subsample_{SUBSAMPLE_SIZE}_pairdf.csv"
    x1, y1, feature_name1 = get_data(
        'income', state1, True, './data/acs/', 2018
        )
    x2, y2, _ = get_data(
        'income', state2, True, './data'
        )
    X1_processed, X2_processed, feature_names_processed = preprocess_whyshift(
        x1, x2, feature_names1
    )
    #Checking whether to subsampling here:
    n1 =  min(subsample_size, X1.shape[0])
    n2 =  min(subsample_size, X2.shape[0])
    idx1 = RNG.choice(X1.shape[0], n1, replace = False)
    idx2 = RNG.choice(X2.shape[0], n2, replace = False)
    X = np.vstack([X1[idx1, :], X2[idx2, :]])
    Y = np.concatenate([y1[idx1], y2[idx2]])
    pair_df = pd.concat([
        pd.DataFrame(X, columns = feature_names1),
        pd.Series(Y, name = 'Y')
    ], axis = 1)
    pair_df.to_csv(out_path, index = False)
    return out_path

'''
Load the whyshift dataset:
'''
def _load_whyshift_pair(state1, state2, pair_path, seed, n_max):
    df_X1, df_Y1, _ = 

'''
Run the tabular benchmark dataset:
'''
def run_tabular_benchmark(shift_type = 'adv', completed = None):
    if completed is None:
        completed = load_completed_ids(SCORES_CSV)


'''
Impose linear shift on the whyshift dataset here - 
only impose the linear concept drift for the whyshift data.
'''
def impose_linear_cd_whyshift(df1_X, df1_Y, df2_X, df2_Y, n_shift):
    beta1 = np.zeros(df1_X.shape[1])
    beta1[:n_shift] = np.linspace(1.5, 0.4, n_shift)
    df2_Y = df2_Y + df2_X @ beta1
    df1 = np.hstack([df1_X, df1_Y.reshape(-1, 1)])
    df2 = np.hstack([df2_X, df2_Y.reshape(-1, 1)])
    return df1, df2, np.arange(n_shift, dtype=int)



'''
Load the pairs of datasets in the whyshift dataset:
'''
def load_whyshift_pair(state1, state2, pair_path, max_sample, seed = 2025):
    #extract the whyshift paired data:
    rng = np.random.default_rng(seed)
    df_X1, df_Y1, _ = get_data('income', state1, True, "./dataset/acs/", 2018)
    df_X2, df_Y2, _ = get_data('income', state2, True, "./dataset/acs/", 2018)
    df1 = np.hstack([df1_X, df1_Y.reshape(-1, 1)])
    df2 = np.hstack([df2_X, df2_Y.reshape(-1, 1)])
    if df1.shape[0] > max_sample:
        df1 = df1[rng.choice(df1.shape[0], n_max, replace = False)]
    if df2.shape[0] > max_sample:
        df2 = df2[rng.choice(df2.shape[0], n_max, replace = False)]    
    df1_X = df1[:, :-1]
    df1_Y = df1[:, -1].ravel()
    df2_X = df2[:, :-1]
    df2_Y = df2[:, -1].ravel()
    return df1_X, df1_Y, df2_X, df2_Y


'''
transforming the results to the rows - 
the columns are the outputs: scenario, dataset_id etc.
'''
def save_result(
    scenario, dataset_id,
    shift_type, result, 
    scores_csv, metrics_csv,
    rep = 0
):
    rows_scores = []
    rows_metrics = []
    feature_ind = np.asarray(result['feature_ind'], dtype = int)
    for method, scores in result['scores'].items():
        scores = np.asarray(scores, dtype = float).ravel()
        ranks = np.argsort(-scores)
        for feat_idx, (scores, rank) in enumerate(zip(scores, ranks)):
            rows_scores.append({
                'scenario': scenario,
                'dataset_id': dataset_id,
                'shift_type': shift_type,
                'rep': rep,
                'method': method,
                'feature': int(feat_idx),
                'score': float(score),
                'rank': int(rank),
                'is_shift_feature': int(feat_idx in shift_set)
            })
        m = result['metrics'][method]
        row = {
          'scenario': scenario,
          'dataset_id': dataset_id,
          'shift_type': shift_type,
          'rep': rep,
          'method': method,
          'n_shift_features': int(len(feature_ind))
        }
        row.update(m)
        rows_metrics.append(row)
    rows_score_df = pd.DataFrame(rows_scores).to_csv(scores_csv,
        index = False)
    rows_metrics_df = pd.DataFrame(rows_metrics).to_csv(metrics_csv,
        index = False)
    return rows_score_df, rows_metrics_df


#save the json results to the csv file:
def save_json_result(json_path, key, result):
    data = {}
    if os.path.exists(json_path):
        with open(json_path, 'r', encoding = 'utf-8') as f:
            data = json.load(f)
    data[str(key)] = result['metrics']
    with open(json_path, 'w', encoding = 'utf-8') as f:
        json.dump(data, f, indent = 2)













