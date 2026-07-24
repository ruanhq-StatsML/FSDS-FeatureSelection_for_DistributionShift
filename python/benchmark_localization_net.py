"""FSL-Net and DataFix (DF-Locate) feature localization benchmarks."""
import os
import sys
import time
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from datafix import DFLocate
from realdata_vimp_benchmark import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, roc_auc_score

def metrics_from_threshold(scores, feature_ind, p, threshold):
    scores = np.asarray(scores, dtype=float).ravel()
    hard = scores > threshold
    selected = np.where(hard)[0]
    return metrics(selected, feature_ind, p, scores=scores)

def metrics_from_mask(scores, mask, feature_ind, p):
    scores = np.asarray(scores, dtype=float).ravel()
    selected = np.where(np.asarray(mask, dtype=int) == 1)[0]
    return metrics(selected, feature_ind, p, scores=scores)

def benchmark_localization_net(
    df1,
    df2,
    feature_ind,
    threshold=0.75,
    cv_fold=5,
    B=1,
    random_state=0,
    methods=("datafix", "fslnet"),
):
    """
    Benchmark DataFix DF-Locate vs FSL-Net on reference/query feature matrices,
    leverage the code from the 
    https://github.com/AI-sandbox/FSL-Net
    https://github.com/AI-sandbox/DataFix
    --------
    df1, df2 : array-like
        Feature matrices (n_samples, n_features). No response column.
    feature_ind : array-like
        Ground-truth shifted feature indices.
    B : int
        Number of repeated runs (DataFix refit each time; FSL-Net is deterministic).
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    df1 = np.asarray(df1, dtype=np.float32)
    df2 = np.asarray(df2, dtype=np.float32)
    feature_ind = np.asarray(feature_ind, dtype=int).ravel()
    n1, p = df1.shape
    n2 = df2.shape[0]
    gt_ind = np.zeros(p, dtype=int)
    gt_ind[feature_ind] = 1
    methods = tuple(methods)
    run_datafix = "datafix" in methods
    run_fslnet = "fslnet" in methods
    datafix_scores = np.zeros((B, p), dtype=float)
    datafix_masks = np.zeros((B, p), dtype=int)
    fslnet_scores = np.zeros((B, p), dtype=float)
    ts_datafix = np.zeros(B, dtype=float)
    ts_fslnet = np.zeros(B, dtype=float)
    f1_datafix = np.zeros(B, dtype=float)
    f1_fslnet = np.zeros(B, dtype=float)
    auc_datafix = np.zeros(B, dtype=float)
    auc_fslnet = np.zeros(B, dtype=float)
    # Run DataFix before loading PyTorch/FSL-Net (fork after torch deadlocks on macOS).
    for b in range(B):
        if run_datafix:
            t0 = time.perf_counter()
            locator = DFLocate(
                estimator=RandomForestClassifier(
                    max_depth=5,
                    n_estimators=150,
                    min_samples_leaf=max(1, round(np.sqrt(n1 + n2) // 2)),
                    max_features="sqrt",
                    random_state=random_state + b,
                ),
                cv=cv_fold,
                test_size=None,
                random_state=random_state + b,
                verbose=False,
                n_jobs=-1,
            )
            locator.shift_location(df1, df2)
            scores = np.asarray(locator.importances_, dtype=float).ravel()
            mask = np.asarray(locator.mask_, dtype=int).ravel()
            datafix_scores[b] = scores
            datafix_masks[b] = mask
            f1_datafix[b] = f1_score(gt_ind, mask, zero_division=0)
            auc_datafix[b] = roc_auc_score(gt_ind, scores)
            ts_datafix[b] = time.perf_counter() - t0
    if run_fslnet:
        dev = _resolve_device(device)
        from fslnet.fslnet import FSLNet
        fsl_model = _get_fsl_model(dev)
        ref_t = torch.tensor(df1, dtype=torch.float32, device=dev)
        que_t = torch.tensor(df2, dtype=torch.float32, device=dev)
        for b in range(B):
            t0 = time.perf_counter()
            with torch.no_grad():
                soft_pred, _ = fsl_model(ref_t, que_t)
            scores = soft_pred.detach().cpu().numpy().ravel()
            hard = scores > threshold
            fslnet_scores[b] = scores
            f1_fslnet[b] = f1_score(gt_ind, hard, zero_division=0)
            auc_fslnet[b] = roc_auc_score(gt_ind, scores)
            ts_fslnet[b] = time.perf_counter() - t0
    result = {
        "scores": {},
        "metrics": {},
        "summary": {},
        "time_profile": {},
        "feature_ind": feature_ind.tolist(),
        "p": int(p),
    }
    if run_datafix:
        mean_datafix = datafix_scores.mean(axis=0)
        mean_mask = (datafix_masks.mean(axis=0) >= 0.5).astype(int)
        result["scores"]["datafix"] = mean_datafix
        result["metrics"]["datafix"] =  metrics_from_mask(
            mean_datafix, mean_mask, feature_ind, p
        )
        result["summary"]["datafix"] = {
            "avgAUC": float(np.nanmean(auc_datafix)),
            "stdAUC": float(np.nanstd(auc_datafix)),
            "avgF1": float(np.mean(f1_datafix)),
            "stdF1": float(np.std(f1_datafix)),
            "meanTS": float(np.mean(ts_datafix)),
            "stdTS": float(np.std(ts_datafix)),
        }
        result["time_profile"]["datafix"] = float(np.mean(ts_datafix))
    if run_fslnet:
        mean_fslnet = fslnet_scores.mean(axis=0)
        result["scores"]["fslnet"] = mean_fslnet
        result["metrics"]["fslnet"] = metrics_from_threshold(
            mean_fslnet, feature_ind, p, threshold
        )
        result["summary"]["fslnet"] = {
            "avgAUC": float(np.nanmean(auc_fslnet)),
            "stdAUC": float(np.nanstd(auc_fslnet)),
            "avgF1": float(np.mean(f1_fslnet)),
            "stdF1": float(np.std(f1_fslnet)),
            "meanTS": float(np.mean(ts_fslnet)),
            "stdTS": float(np.std(ts_fslnet)),
        }
        result["time_profile"]["fslnet"] = float(np.mean(ts_fslnet))
    return result



