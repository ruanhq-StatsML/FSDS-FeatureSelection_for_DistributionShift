#LOCO r-risk variable importance:
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from joblib import Parallel, delayed, parallel_backend
from dataclasses import dataclass
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from scipy.special import kl_div
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from benchmark_config import MODEL_REGISTRY
from model_registry_class import *
from typing import Any, Dict, Literal, Optional, Sequence, Tuple
from joblib import Parallel
from VIMP_DS_utils import (_as_1d, _as_2d,
 _rng_seed, 
 _as_1d_float, _as_2d_float,
 _standardize_cols, _safe_normalize,
 infer_response_type
)


os.chdir("/Users/heqiaoruan/Library/Mobile Documents/com~apple~CloudDocs/Documents/GitHub 2/Causal_Objective_Permutation_Test/Python") 


"""
#R-learner weighted regression with those inputs:
"""
def _fit_tau_rlearner_weighted(X, Y_tilde, W_tilde, seed: int = 0, clip_wtilde: float = 1e-3):
    X = np.asarray(X)
    Y_tilde = np.asarray(Y_tilde).reshape(-1)
    W_tilde = np.asarray(W_tilde).reshape(-1)
    mask = (np.abs(W_tilde) > clip_wtilde)
    z = Y_tilde[mask]/W_tilde[mask]
    weights = (W_tilde[mask] ** 2)
    tau_model = Pipeline(
      steps = [
      ('scaler', StandardScaler(with_mean = True, with_std = True)),
      ('ridge', Ridge(alpha = 1.0, random_state = seed))])
    tau_model.fit(X[mask], z, ridge__sample_weight = weights)
    return tau_model


'''
Making the folds here via the array_split with the 5 folds to split.
'''
def make_folds(n, n_folds = 5, seed = 2026):
    rng = np.random.default_rng(seed)
    indices = np.arange(n)
    rng.shuffle(indices)
    folds = np.array_split(indices, n_folds)
    return folds


'''
Cross-fitting for the outcome model and the propensity score model:
'''
def cross_nuisance_fit(X, Y, W, n_folds = 5,
  n_estimators = 100, binary_outcome = True,
  model_registry = MODEL_REGISTRY,
  model_m = 'rf_regressor', model_e = 'rf_classifier',
  clip_e = 1e-3, seed = 2026):
    X = np.asarray(X)
    Y = _as_1d(Y)
    W = _as_1d(W).astype(int)
    n = X.shape[0]
    rng = np.random.default_rng(seed)
    folds = make_folds(n, n_folds = n_folds, seed = seed)
    mu_hat = np.zeros(n, dtype = float)
    e_hat = np.zeros(n, dtype = float)
    for k, test_idx in enumerate(folds):
        train_idx = np.setdiff1d(np.arange(n), test_idx)
        X_train, X_test = X[train_idx], X[test_idx]
        Y_train = Y[train_idx]
        W_train = W[train_idx]
        model_propensity_score = model_registry[model_e]
        model_outcome = model_registry[model_m]
        #Fit the outcome model and propensity score models:
        fit_mu = model_outcome['fit'](
          X_train, Y_train, seed = seed + k
          )
        mu_hat[test_idx] = model_outcome['predict'](
          fit_mu, X_test
          )
        fit_e = model_propensity_score['fit'](
          X_train, W_train, seed = seed + k * 3
          )
        e_hat[test_idx] = model_propensity_score['predict'](
          fit_e, X_test
          )
    e_hat = np.clip(e_hat, clip_e, 1 - clip_e)
    return mu_hat, e_hat




#With the online anomaly detection procedure here for enhancements!
DEFAULT_MODEL_FACTORY = ModelRegistry(
  ntree = 150,
  ridge_alpha = 0.25,
  nthread = 1, 
  maxit = 200,
  max_depth = 5,
  gamma = 0.25,
  eta = 0.15, 
  mlp_hidden_size = 4,
  mlp_decay = 1e-5, 
  mlp_max_iter = 500, 
  mlp_trace = False,
  mlp_max_coef_reg = 10000, 
  mlp_max_coef_clf = 10000,
  warn_xgb_labels = True, 
  positive_class = 1
)

MODEL_REGISTRY = DEFAULT_MODEL_FACTORY.as_r_style_dict()
def vimp_loco_r_risk(X, Y, W, model_registry = MODEL_REGISTRY,
    model_m  ='rf_regressor', model_e = 'logistic_classifier',
    seed: int = 2026, n_folds = 5,
    clip_e = 0.01):
    X = np.asarray(X)
    Y = _as_1d(Y)
    W = _as_1d(W).astype(int)
    n = X.shape[0]
    folds = make_folds(n, n_folds = n_folds, seed = seed)
    mu_hat, e_hat = cross_nuisance_fit(
        X, Y, W, n_folds = n_folds, model_registry = MODEL_REGISTRY,
        model_m = model_m, model_e = model_e)
    e_hat = np.clip(e_hat, clip_e, 1 - clip_e)
    Y_tilde = Y - mu_hat
    W_tilde = W - e_hat
    tau_model = _fit_tau_rlearner_weighted(X, Y_tilde, W_tilde, seed = seed)
    tau_hat = tau_model.predict(X)
    observed_r_risk = np.mean((Y_tilde - tau_hat * W_tilde) ** 2)
    k = X.shape[1]
    #LOCO variable importance here, you need to refit the outcome model and propensity score modeo right?
    #Can be parallel:
    def calculate_individual_LOCO(i, X, Y, W, n_folds = n_folds,
        model_registry = MODEL_REGISTRY,
        model_m = model_m, 
        model_e = model_e, clip_e = 0.01):
        p = X.shape[1]
        seed = 2000 + i
        X_loco = X[:, np.setdiff1d(np.arange(p), i)]
        mu_hat_i, e_hat_i = cross_nuisance_fit(
            X_loco, Y, W, n_folds = n_folds, model_registry = MODEL_REGISTRY,
            model_m = model_m, model_e = model_e
        )
        e_hat_i = np.clip(e_hat_i, clip_e, 1 - clip_e)
        Y_tilde_i = Y - mu_hat_i
        W_tilde_i = W - e_hat_i
        tau_model_i = _fit_tau_rlearner_weighted(X_loco, Y_tilde_i, W_tilde_i, seed = seed)
        tau_hat_i = tau_model_i.predict(X_loco)
        rrisk_dif = np.mean((Y_tilde_i - tau_hat_i * W_tilde_i) ** 2)
        return rrisk_dif
    with parallel_backend('threading', n_jobs = -1):
        loco_rrisk_list = Parallel(n_jobs = -1)(
            delayed(calculate_individual_LOCO)(i, X, Y, W, 
            n_folds = n_folds, model_registry = MODEL_REGISTRY,
            model_m = model_m, model_e = model_e) for i in range(k)
        )
    #output the LOCO VIMP for the R-risk and their corresponding rankings:
    LOCO_RRisk_VIMP = loco_rrisk_list - observed_r_risk
    loco_rrisk_rank = np.argsort(-LOCO_RRisk_VIMP)
    return {
    'LOCO_VIMP_R': LOCO_RRisk_VIMP,
    'LOCO_VIMP_Rank_R': loco_rrisk_rank
    }
