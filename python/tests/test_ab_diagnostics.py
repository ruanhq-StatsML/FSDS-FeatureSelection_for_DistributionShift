import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ab_diagnostics import (  # noqa: E402
    aipw_ate,
    block_loco_mmd,
    dependence_shift_test,
    modality_shift_decomposition,
    split_balance_c2st,
    stage_shift_report,
)
from demo_ab_diagnostics import MODALITY_BLOCKS, simulate_ab, simulate_cross_modal, simulate_modalities  # noqa: E402

TEXT_IMAGE = frozenset(["text", "image"])


def test_balanced_split_is_not_flagged():
    X, _, W = simulate_ab(n=4000, seed=3)
    rep = split_balance_c2st(X[W == 0], X[W == 1], n_perm=200, seed=3)
    assert rep.p_value > 0.01


def test_broken_split_is_flagged_and_localised():
    X, _, W = simulate_ab(n=6000, broken_split=True, seed=4)
    rep = split_balance_c2st(X[W == 0], X[W == 1], n_perm=200, seed=4)
    assert rep.p_value < 0.05
    assert rep.top(1)[0][0] == "x2"


def test_aipw_beats_diff_in_means_and_covers_truth():
    X, Y, W = simulate_ab(n=6000, true_ate=0.05, seed=5)
    res = aipw_ate(X, Y, W, propensity=0.5, pre_period_col=0, n_folds=3, seed=5)
    assert res.se < res.diff_in_means_se
    assert res.ci[0] - 0.02 < 0.05 < res.ci[1] + 0.02


def test_cross_modal_shift_invisible_to_marginals():
    ref, new, blocks = simulate_cross_modal(n=800, seed=6)
    out = block_loco_mmd(ref, new, blocks, max_n=800, seed=6)
    assert out["__full__"]["loco"] > 0.01
    for name in blocks:
        assert out[name]["marginal"] < 0.25 * out["__full__"]["loco"]


def test_stage_concept_drift_localised():
    rng = np.random.default_rng(7)
    X1, X2 = rng.normal(size=(2000, 5)), rng.normal(size=(2000, 5))
    Y1 = X1[:, 0] + rng.normal(size=2000)
    Y2 = X2[:, 0] + 0.8 * X2[:, 3] + rng.normal(size=2000)
    rep = stage_shift_report(X1, Y1, X2, Y2, seed=7)
    assert rep["covariate"].p_value > 0.01
    assert rep["concept"].heterogeneity_p_value < 0.01
    assert rep["concept"].top(1)[0][0] == "x3"


def test_redundant_shift_is_masked_by_lomo_but_not_by_lomi():
    ref, new = simulate_modalities("redundant", n=600, seed=8)
    r = modality_shift_decomposition(ref, new, MODALITY_BLOCKS, max_n=600, n_perm=100, seed=8)
    for j in ("text", "image"):
        assert r.p_lomi[j] < 0.05
        assert r.lomo[j] < 0.25 * r.lomi[j]
    assert r.interaction[TEXT_IMAGE] < 0


def test_dependence_flip_is_synergistic():
    ref, new = simulate_modalities("dependence_flip", n=600, seed=9)
    r = modality_shift_decomposition(ref, new, MODALITY_BLOCKS, max_n=600, n_perm=100, seed=9)
    assert r.p_joint < 0.05
    for j in ("text", "image"):
        assert r.p_lomi[j] > 0.01
        assert r.lomo[j] > 0
    assert r.interaction[TEXT_IMAGE] > 0
    assert dependence_shift_test(ref, new, MODALITY_BLOCKS, max_n=600, n_perm=100, seed=9)["p_value"] < 0.05


def test_dependence_test_ignores_marginal_shift():
    for scenario in ("unique_text", "redundant"):
        ref, new = simulate_modalities(scenario, n=600, seed=10)
        assert dependence_shift_test(ref, new, MODALITY_BLOCKS, max_n=600, n_perm=100, seed=10)["p_value"] > 0.01


def test_shapley_is_efficient():
    ref, new = simulate_modalities("unique_text", n=400, seed=11)
    r = modality_shift_decomposition(ref, new, MODALITY_BLOCKS, max_n=400, n_perm=20, seed=11)
    assert abs(sum(r.shapley.values()) - r.value[frozenset(MODALITY_BLOCKS)]) < 1e-9
