import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from domain_classifier import compute_dc_vimp_batches  # noqa: E402
from LOCO_vimp_r_risk import vimp_loco_r_risk  # noqa: E402
from permuCATE_vimp import permuCATE_vimp  # noqa: E402


def _hte_data(n=1200, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 5))
    W = rng.binomial(1, 0.5, n)
    Y = X[:, 0] + W * (1 + 2 * X[:, 2]) + rng.normal(size=n)
    return X, Y, W


def test_permucate_ranks_effect_modifier_not_prognostic_feature():
    X, Y, W = _hte_data()
    v = permuCATE_vimp(X, Y, W, model_e="rf_classifier", n_perm=5, seed=0)
    assert np.argmax(v) == 2
    assert v[0] < 0.1 * v[2]


def test_permucate_handles_categorical_features():
    rng = np.random.default_rng(1)
    n = 1200
    X = rng.normal(size=(n, 4))
    X[:, 3] = rng.integers(0, 3, n)
    W = rng.binomial(1, 0.5, n)
    Y = X[:, 0] + W * (1 + 1.5 * (X[:, 3] == 2)) + rng.normal(size=n)
    v = permuCATE_vimp(X, Y, W, model_e="rf_classifier", n_perm=5, seed=0)
    assert np.argmax(v) == 3


def test_loco_r_risk_runs_and_ranks_effect_modifier_first():
    X, Y, W = _hte_data(n=1000, seed=2)
    r = vimp_loco_r_risk(X, Y, W, model_e="rf_classifier", n_folds=3, seed=0)
    assert r["LOCO_VIMP_R"].shape == (5,)
    assert r["LOCO_VIMP_Rank_R"][0] == 2


def test_domain_classifier_vimp_all_modes():
    rng = np.random.default_rng(3)
    X1 = rng.normal(size=(800, 5))
    X2 = rng.normal(size=(800, 5))
    X2[:, 1] += 0.8
    for vimp_type in ("LOCO", "condperm", "oob"):
        _, rank = compute_dc_vimp_batches(X1, X2, model="logistic", vimp_type=vimp_type, seed=0, n_perm_reps=3)
        assert rank[0] == 1, vimp_type
