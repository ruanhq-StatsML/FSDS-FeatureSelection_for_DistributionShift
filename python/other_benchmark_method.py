#The list of a
import os
os.chdir(
    "/Users/heqiaoruan/Documents/GitHub 2/Causal_Objective_Permutation_Test/Python"
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
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from domain_classifier_VIMP_whole import compute_vimp_from_adapter
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from permuCATE_vimp import permuCATE_vimp
from grf_vimp_causalForest import cf_variable_importance
from VIMP_drperm_benchmark import DRPerm_LOCO
from LOCO_vimp_r_risk import vimp_loco_r_risk


'''

df1_X = np.random.random((200, 19))
df2_X = np.random.random((200, 19))
df1_Y = np.random.random((200, 1))
df2_Y = np.random.random((200, 1))
X = np.vstack([df1_X, df2_X])
Y = np.concatenate([df1_Y, df2_Y])
W = np.concatenate([np.zeros(200), np.ones(200)])

'''


'''
Conductin the PCA on the shifted pairs:
Inputs:
arr1:            Input a dataframe or array with size (n, p)
arr2:            Input a dataframe or array with size (n, p)
n_pc:            Number of extracted PC(principal components)
subsample:       The subsampling size here.
seed:            Seed
'''
def pca_covariate_shift_pair(
    arr1, arr2,
    n_pc = 30, random_state = 2010,
    subsample = 2000, seed = 2024):
    from sklearn.decomposition import PCA
    X1 = _subsample_rows(arr1, subsample, seed = seed)
    X2 = _subsample_rows(arr2, subsample, seed = seed * 2)
    n_comp = adaptive_pca_n(X1.shape[0], X2.shape[0], 
        X1.shape[1], target_pca)
    pca = PCA(n_components = n_comp, random_state = random_state)
    X1_p = pca.fit_transform(X1)
    pca = PCA(n_components = n_comp, random_state = random_state)    
    X2_p = pca.fit_transform(X2)
    n1, n2 = X1_p.shape[0], X2_p.shape[0]
    y1 = np.arange(n1, dtype = float)
    y2 = np.arange(n2, dtype = float)
    out1 = np.hstack([X1_p, y1.reshape(-1, 1)])
    out2 = np.hstack([X2_p, y2.reshape(-1, 1)])
    return out1, out2


'''
Computing the delta shapley value:
Inputs:
df1_X:      Input a dataframe or array with size (n, p)
df1_Y:      Input a dataframe or array with size (n, )
df2_X:      Input a dataframe or array with size (n, p)
df2_Y:      Input a dataframe or array with size (n, )
test_size:  The proportion of the test size ranging from 0 to 1.
seed:       The random seed, by default.

Return:
delta_shap: The variable importance values.
rank:       The rank of the features extracted
'''
def compute_delta_shap(
    df1_X, df1_Y,
    df2_X, df2_Y,
    test_size = 0.25, seed = 2025):
    rng = np.random.default_rng(seed)
    p = df1_X.shape[1]
    n1 = df1_X.shape[0]
    n2 = df2_X.shape[0]
    rf_params = dict(
        max_depth = max(2, round(p // 3)),
        n_estimators = 100,
        min_samples_leaf = max(1, round(np.sqrt(n1+n2)//2)),
        n_jobs = 1,
        random_state = seed
    )
    model1 = RandomForestRegressor(**rf_params)
    model2 = RandomForestRegressor(**{**rf_params, 'random_state': seed + 1})
    X_train1, X_test1, Y_train1, _ = train_test_split(
        df1_X, df1_Y, test_size = test_size, random_state = seed + 10
    )
    X_train2, X_test2, Y_train2, _ = train_test_split(
        df2_X, df2_Y, test_size = test_size, random_state = seed + 12
    )
    model1.fit(X_train1, Y_train1)
    model2.fit(X_train2, Y_train2)
    explainer1 = shap.Explainer(model1, X_train1)
    shap_value_b1 = explainer1(X_test1, check_additivity = False)
    explainer2 = shap.Explainer(model2, X_train2)
    shap_value_b2 = explainer2(X_test2, check_additivity = False)
    mean_shap_b1 = np.mean(shap_value_b1.values, axis = 0)
    mean_shap_b2 = np.mean(shap_value_b2.values, axis = 0)
    delta_shap = np.abs(mean_shap_b2 - mean_shap_b1)
    return delta_shap, np.argsort(-delta_shap)




'''
Implementation for the random forest classifier:
Input:
df1_X:     The covariates in the existing batch of data
df2_X:     The covariates in the new batch of data
seed:      Random Seed

Return:
variable importance values and the variable importance rankings for the classifier
'''
def rf_domain_classifier(df1_X, df2_X, seed = 0):
    X = np.vstack([df1_X, df2_X])
    Y = np.concatenate([np.zeros(df1_X.shape[0]),
                        np.ones(df2_X.shape[0])])
    n, p = X.shape
    #Leverage the default hyperparameter setting for the random forest binary classifier,
    #mtry = sqrt(p) and min_node_size = sqrt(n)/2
    rf_model = RandomForestClassifier(
        n_estimators = 150,
        max_depth = 5,
        max_features = round(np.sqrt(p)/p, 3),
        max_node_size = round(np.sqrt(n)/2)
    )
    rf_model.fit(X, Y)#benchmark the random forest classifier here with the variable importance.
    vimp_list = rf_model.feature_importances_
    vimp_rank = np.argsort(-np.asarray(vimp_list, dtype = float))
    return vimp_list, vimp_rank


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
Benchmark the SGShift methodology for the linear/nonlinear feature selection,
leveraged the code from "https://openreview.net/forum?id=lSlXOD2v40"

Return the statistics for each of the method:
[1] L1-specification procedure
[2] L1-misspecified model procedure
[3] Knockoff based procedure.

Return:
The dictionary containing the following:
l1_score_spec:       The VIMP score for the l1 specified procedure.
l1_selected_spec:    The selected features for l1 specified procedure.
l1_score_misspec:    The VIMP score for the l1 misspecified procedure.
l1_selected_misspec: The selected features for l1 misspecified procedure.
knock_score:         The VIMP score for the knockoff procedure.
selected_knock:      The selected features for the knockoff procedure.
'''
def sgs_shift_benchmark(
  df1_X, df1_Y,
  df2_X, df2_Y,
  task = 'regression',
  model_type = 'random_forest_reg',
  selection_tol = 1e-4,
  random_state = 24,
  lambdas = None,
  v = 3, B = 20,
  pi_thr = 0.5):
    X_S = np.asarray(df1_X, dtype = float)
    X_T = np.asarray(df2_X, dtype = float)
    y_S = np.asarray(df1_Y)
    y_T = np.asarray(df2_Y)
    if task == 'classification':
        y_S = y_S.astype(int)
        y_T = y_T.astype(int)
    else:
        y_S = y_S.astype(float)
        y_T = y_T.astype(float)
    if lambdas is None:
        lambdas = list(np.logspace(-2.5, 0.3, 8))    
    solver = SGShift.fit_model(task, model_type, random_state, X_S, y_S)
    _, delta_mis, _ = SGShift.cross_validate_lambda(
        X_T,
        y_T,
        solver,
        lambdas,
        misspecified=True,
        X_S=X_S,
        y_S=y_S,
        task=task,
    )
    scores_l1mis = np.abs(delta_mis)
    selected_l1mis = np.where(scores_l1mis > selection_tol)[0]
    delta_unmis, _ = SGShift.cross_validate_lambda(
        X_T,
        y_T,
        solver,
        lambdas,
        misspecified=False,
        X_S=X_S,
        y_S=y_S,
        task=task,
    )
    scores_l1unmis = np.abs(delta_unmis)
    selected_l1spec = np.where(scores_l1unmis > selection_tol)[0]
    pi = SGShift.derandom_knock(
        solver, X_S, y_S, X_T, y_T, B, lambdas, v=v, task=task
    )
    freq = np.max(np.vstack([pi[l] for l in lambdas]), axis=0)
    selected_knock = np.where(freq >= pi_thr)[0]
    return {
        "l1_score_spec": scores_l1unmis,
        "l1_selected_spec": selected_l1spec,
        "l1_score_misspec": scores_l1mis,
        "l1_selected_misspec": selected_l1mis,
        "knock_score": freq,
        "selected_knock": selected_knock,
    }


'''
Calculate the feature selection performances:
Inputs:
selected:      The array containing the selected indices of features.
shift_index:   The array containing the ground-truth indices for the shifted features.
p:             The number of features.
scores:        The VIMP scores to record.
Outputs:
The dictionary returning 
TP(True Positive Count), FP(False Positive Count), FN(False Negative Count),
Precision, Recall, F1-score and the FDR(False Discovery Rate), 
Selection_AUC(if the score is available)

'''
def metrics(selected, shift_index, p, scores=None):
    k = len(selected)
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
    if scores is not None and 0 < length(list(true)) < p:
        out["selection_AUC"] = round(float(roc_auc_score(membership, sc)), 4)
    return out


def metrics_from_scores(scores, feature_ind, p):
    k = len(feature_ind)
    selected = np.argsort(-np.asarray(scores, dtype=float))[:k]
    return metrics(selected, feature_ind, p, scores=scores)



'''
Functions for the causal inference variable importances:
PermuCATE, LOCO-R-Risk and the Causal Forest Variable Importance
Inputs:
The X, Y, W are concatenated for both batches of data
X:            The covariates, shaped (n1 + n2, p)
Y:            The response, shaped (n1 + n2, )
W:            The batch assignment labels, shaped (n1 + n2, )
model_m:      The outcome model, by default 'rf_regressor'.
model_e:      The propensity score model, by default 'rf_classifier'.
model_tau:    The pseudo-outcome regression model, by default 'rf_regressor'.
model_nu:     The conditional dependence model, by defautl 'rf_regressor'.
n_estimators: The number of trees required.
n_perm:       The number of permutations.
seed:         The seed
Outputs:
perm_VIMP, loco_vimp, cf_vimp are all arrays for variable importance score 
with shape (p, )
'''
def run_all_causal_vimp(X, Y, W,
    model_m = 'rf_regressor',
    model_e = 'rf_classifier',
    model_tau = 'rf_regressor',
    model_nu = 'rf_regressor',
    n_estimators = 150,
    n_perm = 25,
    seed = 2025):
    perm_VIMP = permuCATE_vimp(
        X, Y, W, 
        model_m = model_m, model_e = model_e, seed = seed)
    cf_vimp = cf_variable_importance(
        X, Y, W, 
        model_psm = 'logistic',
        model_outcome = 'rf',
        n_estimators = 100,
        seed = seed
    )
    loco_vimp = vimp_loco_r_risk(X, Y, W,
        model_m = model_m, model_e = model_e, seed = seed)
    return {
        'permucate': perm_VIMP,
        'loco_r_risk': loco_vimp,
        'grf': cf_vimp
    }









'''
Binary Classification Model for the feature selection leveraging the variable importances
The versions include LOCO, Boruta, vitaPIMP, shadowVIMP, condperm, oob.
- adapter:      The model directory, by default, random forest regressor
- fitted_obj:   The fitted model objective.
- X_train:      The training covariates.
- Y_train:      The training responses.
- X_test:       The testing covariates.
- Y_test:       The testing responses.
- vimp_type:    LOCO variable importance.
- n_classes:    Number of classes, by default binary.
- random_state: The random state.
- n_perm_reps:  Number of permutations.
- oob_config:   The configurations for the OOB procedure.
'''
def compute_vimp_from_adapter(
    adapter, fitted_obj,
    X_train, Y_train,
    X_test, Y_test, vimp_type = 'LOCO',
    n_classes = 2, random_state = 42,
    n_perm_reps = 10,
    oob_config: Optional[OOBVIMPConfig] = None,
):
    n, p = X_test.shape
    predictions_test = adapter['predict'](fitted_obj, X_test)
    if predictions_test.ndim == 1:
        predictions_test = predictions_test.reshape(-1, 1)
    #Getting the One-Hot results and return to the Overall L2-loss:
    n_classes = len(set(Y_train).union(set(Y_test)))
    Y_onehot = np.zeros((n, n_classes), dtype = np.float64)
    Y_onehot[np.arange(n), Y_test] = 1.0#The brier scores.
    baseline_risk = np.mean((predictions_test - Y_onehot) ** 2)
    vimp_scores = np.zeros(p)
    r_df_train = r_df_test = None
    if vimp_type in ('vitaPIMP', 'shadowVIMP'):
        rp = _ensure_rpy2()
        col_x = ['X' + str(i) for i in range(X_train.shape[1])]
        df_train = pd.DataFrame(np.column_stack([X_train, np.asarray(Y_train).ravel()]), columns=col_x + ['Y'])
        df_test = pd.DataFrame(np.column_stack([X_test, np.asarray(Y_test).ravel()]), columns=col_x + ['Y'])
        r_df_train = rp['pandas2ri'].py2rpy(df_train)
        r_df_test = rp['pandas2ri'].py2rpy(df_test)
    if vimp_type == 'LOCO':
        for j in range(p):
            keep_cols = [c for c in range(p) if c != j]
            adapter_new = adapter.copy()
            X_train_j = X_train[:, keep_cols]
            X_test_j  = X_test[:, keep_cols]
            fitted_j = adapter_new['fit'](X_train_j, Y_train, seed = random_state + j)
            prob_j = adapter_new['predict'](fitted_j, X_test_j)
            #It's not necessarily -> to make them aligned..
            if prob_j.ndim == 1:
                prob_j = prob_j.reshape(-1, 1)
            risk_j = np.mean((prob_j - Y_onehot) ** 2)
            vimp_scores[j] = risk_j - baseline_risk
    elif vimp_type == 'condperm':
        # Conditional permutation: average risk increase over n_perm_reps draws.
        rng = np.random.default_rng(seed = random_state)
        n_reps = max(1, int(n_perm_reps))
        for j in range(p):
            rep_scores = np.zeros(n_reps, dtype = np.float64)
            for rep in range(n_reps):
                X_test_perm = X_test.copy()
                X_test_perm[:, j] = rng.permutation(X_test[:, j])
                prob_perm = adapter['predict'](fitted_obj, X_test_perm)
                if prob_perm.ndim == 1:
                    prob_perm = prob_perm.reshape(-1, 1)
                risk_perm = np.mean((prob_perm - Y_onehot) ** 2)
                rep_scores[rep] = risk_perm - baseline_risk
            vimp_scores[j] = float(np.mean(rep_scores))
    #oob: The out-of-bag Variable Importance:
    elif vimp_type == 'oob':
        cfg = oob_config or OOBVIMPConfig()
        oob_mask = _domain_oob_mask(
            X_train, Y_train, seed=random_state, oob_config=cfg
        )
        X_oob = np.asarray(X_train, dtype=float)[oob_mask]
        Y_oob = np.asarray(Y_train, dtype=int).ravel()[oob_mask]
        p = X_oob.shape[1]
        n_classes_oob = len(set(Y_train).union(set(Y_oob)))
        probs_base = adapter['predict'](fitted_obj, X_oob)
        baseline_risk = _brier_risk(Y_oob, probs_base, n_classes_oob)
        rng = np.random.default_rng(seed=random_state)
        n_reps = max(1, int(cfg.n_perm_reps if oob_config is not None else n_perm_reps))
        if oob_config is None and n_perm_reps != cfg.n_perm_reps:
            n_reps = max(1, int(n_perm_reps))
        vimp_scores = np.zeros(p, dtype=np.float64)
        for j in range(p):
            rep_scores = np.zeros(n_reps, dtype=np.float64)
            for rep in range(n_reps):
                X_perm = X_oob.copy()
                X_perm[:, j] = rng.permutation(X_oob[:, j])
                probs_perm = adapter['predict'](fitted_obj, X_perm)
                rep_scores[rep] = _brier_risk(Y_oob, probs_perm, n_classes_oob) - baseline_risk
            vimp_scores[j] = float(np.mean(rep_scores))
    #vitaPIMP: the vita Permutation Variable Importance:
    elif vimp_type == 'vitaPIMP':
        rp = _ensure_rpy2()
        randomForest = rp['randomForest']
        vita = rp['vita']
        stats = rp['stats']
        rf_model = randomForest.randomForest(
            formula=stats.as_formula('Y ~ .'),
            data=r_df_train,
            ntree=150,
            importance=True,
            localImp=False,
        )
        # PIMP permutes the response; use train design + labels.
        x_train_r = r_df_train.rx(True, rp['base'].setdiff(rp['base'].colnames(r_df_train), 'Y'))
        pimp_result = vita.PIMP(
            X=x_train_r,
            y=r_df_train.rx2('Y'),
            rForest=rf_model,
            S=25,
            parallel=False,
        )
        var_imp = pimp_result.rx2('VarImp')
        vimp_scores = np.asarray(var_imp, dtype=float).reshape(-1)
    #shadowVIMP: shadow Variable Importance
    elif vimp_type == 'shadowVIMP':
        rp = _ensure_rpy2()
        r = rp['r']
        base = rp['base']
        r.assign('train_df', r_df_train)
        # Parameter name is num.trees (not num_trees). Keep trees/iters modest for benchmarks, call the rpy models.
        r(
            """
            result <- shadowVIMP::shadow_vimp(
                data = train_df,
                outcome_var = 'Y',
                alphas = c(0.3, 0.1),
                niters = c(15, 25),
                num.threads = 2,
                num.trees = 150
            )
            """
        )
        vimp_result = r['result']
        scores = np.zeros(p, dtype=float)
        extracted = False
        # Preferred: mean permutation importance from first pre_selection step history.
        pre = vimp_result.rx2('pre_selection')
        step1 = pre.rx2('step_1')
        hist = rp['pandas2ri'].rpy2py(base.as_data_frame(step1.rx2('vimp_history')))
        for i in range(p):
            col = f'X{i}'
            if col in hist.columns:
                scores[i] = float(np.nanmean(np.asarray(hist[col], dtype=float)))
                extracted = True
        if not extracted:
            try:
                dec = rp['pandas2ri'].rpy2py(base.as_data_frame(vimp_result.rx2('final_dec_pooled')))
                name_col = None
                for c in dec.columns:
                    if str(c).lower() in ('varname', 'variable', 'feature'):
                        name_col = c
                        break
                score_col = None
                for c in dec.columns:
                    cl = str(c).lower()
                    if cl in ('quantile_pooled', 'p_unadj', 'importance', 'vimp'):
                        score_col = c
                        break
                if name_col is not None and score_col is not None:
                    mapping = dict(zip(dec[name_col].astype(str), dec[score_col].astype(float)))
                    # Higher better: if using p_unadj, invert.
                    invert = str(score_col).lower().startswith('p_')
                    for i in range(p):
                        key = f'X{i}'
                        val = float(mapping.get(key, 0.0))
                        scores[i] = (1.0 - val) if invert else val
                    extracted = True
            except Exception:
                extracted = False
        vimp_scores = scores
    #Boruta procedure here:
    elif vimp_type in ('Boruta', 'boruta'):
        n_fit = int(np.asarray(X_train).shape[0] + np.asarray(X_test).shape[0])
        min_leaf = max(1, int(round(np.sqrt(n_fit) / 2.0)))
        rf_model = RandomForestClassifier(
            n_estimators=150,
            n_jobs=-1,
            class_weight='balanced',
            max_features='sqrt',
            min_samples_leaf=min_leaf,
            random_state=int(random_state),
        )
        feat_selector = BorutaPy(
            rf_model,
            n_estimators='auto',
            verbose=0,
            random_state=int(random_state),
        )
        X_combined = np.vstack([X_train, X_test])
        Y_combined = np.concatenate([np.asarray(Y_train).ravel(), np.asarray(Y_test).ravel()])
        feat_selector.fit(X_combined, Y_combined)
        ranking = np.asarray(feat_selector.ranking_, dtype=float).reshape(-1)
        # Lower Boruta rank = more important; convert to higher-better scores, into the range of 0 and 1 here. 
        vimp_scores = (np.max(ranking) -ranking + 1)/np.max(ranking)
    else:
        raise ValueError(
            f"Please input the correct variable importance: {vimp_type}."
            f"'Only Support vitaPIMP', 'shadowVIMP', or 'Boruta'."
        )
    gc.collect()#remove the cache.
    rank_desc = np.argsort(-vimp_scores)
    return vimp_scores, rank_desc







#Benchmark the whole feature selection procedure:
'''
Input: 
df1:             Existing Batch of Data.
df2:             New Batch of Data.
test_size:       The proportion of the test set in the evaluation set, by default 0.3,
model_registry:  Model registry incorporating all of the types of the models.
feature_ind:     The indices of features that account for the distribution shift significantly.
rfdomain_type:   The types of the methodologies that is incorporated into all of the procedures.
seed:            The random seed we set here.

Return:
result_dict:     The result dictionary for the benchmark feature selection results, including the metrics for each of the method in this procedure.


df1 = np.random.random((100, 21))
df2 = np.random.random((100, 21))
n1, p = df1.shape
n2 = df2.shape
df1 = np.array(df1, dtype = float)
df2 = np.array(df2, dtype = float)
df1_X = df1[:, :(p-1)]
df1_Y = df1[:, p-1].reshape(-1, 1)
df2_X = df2[:, :(p-1)]
df2_Y = df2[:, p-1].reshape(-1, 1)
vimp, rank = compute_vimp_ard_diff(df1_X, df1_Y, df2_X, df2_Y, n_iter = 300)
'''

'''
Split the input dataframe with (X, Y) into 
covariates X and responses Y for both batches.
'''
def split_subset(df1, df2):
    p = df1.shape[1] - 1
    df1_X = np.asarray(df1[:, :p], dtype = float)
    df2_X = np.asarray(df2[:, :p], dtype = float)
    df1_Y = np.asarray(df1[:, p], dtype = float)
    df2_Y = np.asarray(df2[:, p], dtype = float)
    return df1_X, df1_Y, df2_X, df2_Y, p


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
    rf_domain_type = RF_DOMAIN_TYPE
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
        model_m=MODEL_M,
        model_e=MODEL_E,
        model_tau=MODEL_TAU,
        model_nu=MODEL_NU,
        n_perm=N_PERM_CATE,
        seed=seed + 4,
    )
    time_dict["permucate"] = float(time.perf_counter() - t0)
    scores["permucate"] = perm_vimp
    metrics_result["permucate"] = metrics_from_scores(perm_vimp, feature_ind, p)
    t0 = time.perf_counter()
    loco_vimp = vimp_loco_r_risk(X, Y, W, model_m=MODEL_M, model_e=MODEL_E, seed=seed + 4)
    time_dict["loco_r_risk"] = float(time.perf_counter() - t0)
    scores["loco_r_risk"] = loco_vimp
    metrics_result["loco_r_risk"] = metrics_from_scores(loco_vimp, feature_ind, p)
    t0 = time.perf_counter()
    cf_vimp = cf_variable_importance(
        X, Y, W, model_psm="logistic", model_outcome="rf", n_estimators=N_ESTIMATORS, seed=seed + 4
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
    for vimp_type in RFDOMAIN_TYPE:
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







def benchmark_whole_feature_selection(
    df1, df2, rfdomain_type,
    
):
    result = {}
    time_profile = {}
    n1, p = df1.shape
    n2 = df2.shape
    df1 = np.array(df1, dtype = float)
    df2 = np.array(df2, dtype = float)
    df1_X = df1[:, :(p-1)]
    df1_Y = df1[:, p-1].reshape(-1, 1)
    df2_X = df2[:, :(p-1)]
    df2_Y = df2[:, p-1].reshape(-1, 1)
    #Benchmark the adapter with the Random Forest Classifier:
    adapter_RF = RandomForestClassifier(
        max_features = np.sqrt(p)/p,
        n_estimators = 150,
        max_depth = 5,
        min_samples_leaf = max(1, round(np.sqrt(p)//2)),
        n_jobs = 1,
        random_state = seed + 1
    )
    fitted_obj = model_rf['fit'](X_train, Y_train)
    for vimp_type in RFDOMAIN_TYPE:
        time = perf.time_counter()
        vimp_value, vimp_rank = compute_vimp_from_adapter(adapter_RF, fitted_obj,
            X_train, Y_train, X_test, Y_test, vimp_type = vimp_type)
        result[str(vimp_type) + 'RFDomain'] = (vimp_value, vimp_rank)
        time_profile[str(vimp_type) + 'RFDomain'] = perf.time_counter() - time
        #Benchmark the delta shap procedure
        vimp_value, vimp_rank = compute_delta_shap(df1_X, df1_Y, df2_X, df2_Y, )
        #Benchmark the fslnet and datafix procedure:
        result_localization = benchmark_localization_net(df1, df2, feature_ind,
            threshold = 0.7, cv_fold = 5, B = 15)
        result['fslnet'] = result_localization['fslnet']
        result['datafix'] = result_localization['datafix']
        #Benchmark the ARD(automatic relevant discrimination procedure):
        time = perf.time_counter()
        vimp_ard, _ = compute_vimp_ard_diff(df1, df2, n_iter = 300)
        time_profile['ard'] = perf.time_counter() - time
        #Benchmark the random forest domain classifier:
        time = perf.time_counter()
        vimp_value, vimp_rank = compute_vimp_rf_domain(
            np.vstack([df1, df2]),
            np.concatenate([np.zeros(n1), np.ones(n2)])
        )
        time_profile['rfdomain'] = perf.time_counter() - time
        result['rfdomain'] = (vimp_value, vimp_rank)
        #Benchmark the causal inference meta learner methods:
        #Benchmark the MMD LOCO procedure for the variable importance scores:
        time = perf.time_counter()
        vimp_value, vimp_rank = compute_loco_mmd_batches(df1_X, df2_X, max_n = 2000, seed = seed + 6)
        time_profile['loco'] = (perf.time_counter() - time).astype(float)
        vimp_value, vimp_rank = compute_hsic_loco_batches(df1_X, df2_X, max_n = 2000, seed = seed + 5, sigma = 1.0)
        #Benchmark the SGShift procedure for the efficiency comparisons:
        time = perf.time_counter()
        sgs_result = sgs_shift_benchmark(df1_X, df1_Y, df2_X, df2_Y, random_state = seed + 7)
        result['l1_score_misspec'] = sgs_result['l1_score_misspec']
        result['l1_score_spec'] = sgs_result['l1_score_spec']
        result['knock_score'] = sgs_result['knock_score']
        time_profile['sgshift'] = (perf.time_counter() - time).astype(float)
        #Set the input as X(Covariates), Y(Responses) and W(Batch Assignment Vector)
        X = np.vstack([df1_X, df2_X])
        Y = np.concatenate([df1_Y, df2_Y]).reshape(-1, 1)
        W = np.concatenate([np.zeros(n1), np.ones(n2)]).reshape(-1, 1)
        causal_result = run_all_causal_vimp(X, Y, W, seed = seed + 4)
        loco_rank = np.argsort(-causal_result['loco_r_risk'])
        permucate_rank = np.argsort(-causal_result['permucate'])
        grf_rank = np.argsort(-causal_result['grf'])
        result['loco'] = (causal_result['loco_r_risk'], loco_rank)
        result['permucate'] = (causal_result['permucate'], permucate_rank)
        result['grf'] = (causal_result['grf'], grf_rank)
        #Benchmark the DRPerm procedure:
        time = perf.time_counter()
        drperm_result = DRPerm_LOCO(X, Y, W, *,
            model_registry = MODEL_REGISTRY,
            seed = 2026, n_splits = 5)
        result['drperm'] = (drperm_result['po_risk_vimp'],
            drperm_result['po_risk_vimp_rank'])
        time_profile['drperm'] = (perf.time_counter() - time).astype(float)
        #Benchmark the 
        output = {}
        for keys, values in result.items():
            result[key] = selection_AUC












####
#Comparing the efficiency of all of these procedures with the sensitivity of that regions:


def calculate_mCE(results_matrix, baseline_alexnet_errors = None):
    all_accuracies = []
    for c_type, severities in results_matrix.items():
        for severity, acc in severity.items():
            all_accuracies.append(acc)
    mean_accuracy = np.mean(all_accuracies)
    return mean_accuracy































































