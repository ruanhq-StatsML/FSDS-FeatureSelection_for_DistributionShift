"""Weakly-overlapped two-cluster DGP for concept-drift overlap sensitivity."""
import os
import sys
import types
import numpy as np
import pandas as pd
from pathlib import Path
from itertools import product
from repliclust.base import DataGenerator
from repliclust.maxmin.archetype import MaxMinArchetype as Archetype

Archetype, DataGenerator = _bootstrap_repliclust()

OVERLAP_MIXTURE_GRID = np.array(
    [0.01, 0.02, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.75]
)
N1 = 2000
N2 = 2000
P = 30


def _standardize_col(col):
    col = np.asarray(col, dtype=float).ravel()
    return (col - col.mean()) / (col.std() + 1e-8)


def _nonlinear_concept_drift(X, rng, scale):
    half = X.shape[1] // 2
    delta = np.zeros(X.shape[0], dtype=float)
    for j in range(half):
        z = _standardize_col(X[:, j])
        z2 = _standardize_col(X[:, (j + 1) % half])
        delta += float(scale) * (
            rng.uniform(0.5, 1.0) * np.sin(z)
            + rng.uniform(0.25, 0.5) * (z ** 2)
            + rng.uniform(0.2, 0.35) * z * z2
        )
    return delta


def DGP_mixture(n1, n2, p, overlap_mixture=0.2, seed=0):
    """
       Sample two weakly-overlapping populations with nonlinear concept drift.
       We leverage the replicluster package to simulate the arbitrarily weak overlapped
       feature covariate spaces.
    """
    rng = np.random.default_rng(seed)
    archetype = Archetype(
        n_clusters=2,
        dim=p,
        n_samples=200000,
        min_overlap=np.exp(np.log(overlap_mixture) - 0.075),
        max_overlap=np.exp(np.log(overlap_mixture) + 0.075),
    )
    data_generator = DataGenerator(archetype=archetype)
    X, y, _ = data_generator.synthesize(quiet=True)
    y0_ind = np.where(y == 0)[0]
    y1_ind = np.where(y == 1)[0]
    pick0 = rng.choice(y0_ind, n1, replace=False)
    pick1 = rng.choice(y1_ind, n2, replace=False)
    df_X1 = X[pick0, :]
    df_X2 = X[pick1, :]
    beta = np.zeros(p)
    beta[: (p // 2)] = np.linspace(1, 0, p // 2)
    feature_ind = np.arange(p // 2)
    Y1 = (
        df_X1 @ beta
        + _nonlinear_concept_drift(df_X1, rng, scale=0.5)
        + rng.normal(size=n1)
    )
    Y2 = (
        df_X2 @ beta
        + _nonlinear_concept_drift(df_X2, rng, scale=1.0)
        + rng.normal(size=n2)
    )
    df1 = np.hstack([df_X1, Y1.reshape(-1, 1)])
    df2 = np.hstack([df_X2, Y2.reshape(-1, 1)])
    return df1, df2, feature_ind

def DGP_mixture_CS(n1, n2, p, overlap_mixture=0.2, seed=0):
    """
    Sample two weakly-overlapping populations with covariate shift(weak overlapped specified by the overlapping ratio).
    """
    rng = np.random.default_rng(seed)
    archetype = Archetype(
        n_clusters=2,
        dim=p,
        n_samples=200000,
        min_overlap=np.exp(np.log(overlap_mixture) - 0.075),
        max_overlap=np.exp(np.log(overlap_mixture) + 0.075),
    )
    data_generator = DataGenerator(archetype=archetype)
    X, y, _ = data_generator.synthesize(quiet=True)
    y0_ind = np.where(y == 0)[0]
    y1_ind = np.where(y == 1)[0]
    pick0 = rng.choice(y0_ind, n1, replace=False)
    pick1 = rng.choice(y1_ind, n2, replace=False)
    df_X1 = X[pick0, :]
    df_X2 = X[pick1, :]
    beta1 = np.zeros(p)
    beta1[: (p // 2)] = np.linspace(1, 0, p // 2)
    feature_ind = np.arange(p // 2)
    Y1 = df_X1 @ beta1 + rng.normal(size=n1)
    Y2 = df_X2 @ beta1 + rng.normal(size=n2)
    df1 = np.hstack([df_X1, Y1.reshape(-1, 1)])
    df2 = np.hstack([df_X2, Y2.reshape(-1, 1)])
    return df1, df2, feature_ind



#Conduct the sensitivity analysis

OVERLAP_MIXTURE = [0.005, 0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 0.75]
P_SEQ = [10, 20, 30, 50, 75, 100, 150, 200, 300, 500]

for overlap, p in product(OVERLAP_MIXTURE, P_SEQ):
    








