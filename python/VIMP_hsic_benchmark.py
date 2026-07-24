"""LOCO-HSIC utilities (restored for real-data overlap benchmarks)."""
from __future__ import annotations

import numpy as np

from VIMP_mmd_benchmark import _rank_positions, _subsample_batch


def _rbf_kernel(A, B, sigma):
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    A2 = np.sum(A * A, axis=1, keepdims=True)
    B2 = np.sum(B * B, axis=1, keepdims=True)
    dist = A2 + B2.T - 2.0 * (A @ B.T)
    gamma = 1.0 / (2.0 * float(sigma) ** 2 + 1e-12)
    return np.exp(-gamma * np.maximum(dist, 0.0))


def _center_kernel(K):
    n = K.shape[0]
    H = np.eye(n) - np.ones((n, n)) / n
    return H @ K @ H


def _hsic(X, W, sigma_x, sigma_w):
    X = np.asarray(X, dtype=float)
    W = np.asarray(W, dtype=float).reshape(-1, 1)
    n = X.shape[0]
    if n < 3:
        return 0.0
    K = _center_kernel(_rbf_kernel(X, X, sigma_x))
    L = _center_kernel(_rbf_kernel(W, W, sigma_w))
    return float(np.trace(K @ L) / ((n - 1) ** 2))


def compute_hsic_loco_batches(df1_X, df2_X, max_n=1000, seed=0, sigma=1.0):
    rng = np.random.default_rng(seed)
    X1 = _subsample_batch(np.asarray(df1_X, dtype=float), max_n, rng)
    X2 = _subsample_batch(np.asarray(df2_X, dtype=float), max_n, rng)
    X = np.vstack([X1, X2])
    W = np.concatenate([np.zeros(len(X1)), np.ones(len(X2))])
    p = X.shape[1]
    base = _hsic(X, W, sigma_x=sigma, sigma_w=sigma)
    vimp = np.zeros(p, dtype=float)
    for j in range(p):
        X_minus = np.delete(X, j, axis=1)
        vimp[j] = base - _hsic(X_minus, W, sigma_x=sigma, sigma_w=sigma)
    return vimp, _rank_positions(vimp)
