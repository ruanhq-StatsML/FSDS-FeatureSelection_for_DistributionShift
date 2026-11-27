import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from canary_stress import (  # noqa: E402
    PIXEL_CORRUPTIONS,
    build_world,
    fgsm_l2,
    make_batches,
    random_l2,
    run_cell,
)


@pytest.fixture(scope="module")
def world():
    return build_world(seed=0)


@pytest.mark.parametrize("name", list(PIXEL_CORRUPTIONS))
def test_corruptions_keep_shape_and_range(world, name):
    imgs = world.shadow_imgs[:20]
    out = PIXEL_CORRUPTIONS[name](imgs, 3, np.random.default_rng(0))
    assert out.shape == imgs.shape
    assert out.min() >= 0.0 and out.max() <= 1.0
    assert not np.allclose(out, imgs)


def test_fgsm_hurts_more_than_random_noise_of_same_budget(world):
    Z, y = world.embed(world.shadow_imgs), world.shadow_y
    eps = 0.2 * world.z_norm
    acc = lambda A: np.mean(world.clf.predict(A) == y)  # noqa: E731
    Z_adv, Z_rand = fgsm_l2(world, Z, y, eps), random_l2(Z, eps, np.random.default_rng(0))
    assert np.allclose(np.linalg.norm(Z_adv - Z, axis=1), eps)
    assert acc(Z) - acc(Z_adv) > 3 * (acc(Z) - acc(Z_rand)) + 0.1


def test_null_cell_has_no_planted_rows(world):
    _, _, drop = make_batches(world, "none", 1, 0.3, 200, np.random.default_rng(0))
    assert drop == 0.0


def test_strong_corruption_is_detected_and_localised(world):
    res = run_cell(world, "gaussian_blur", 5, 0.3, 300, reps=3, seed=0)
    for m in ("c2st", "mmd_bonferroni"):
        assert res[m]["detect_rate"] == 1.0
        assert res[m]["localised_given_detect"] == 1.0


def test_fgsm_is_invisible_to_input_monitors_but_not_to_output_monitor(world):
    res = run_cell(world, "fgsm_l2", 5, 0.3, 300, reps=3, seed=0)
    assert res["accuracy_drop_on_perturbed"] > 0.5
    assert res["c2st"]["detect_rate"] < 1.0 and res["mmd_bonferroni"]["detect_rate"] < 1.0
    assert res["output_confidence_ks"]["detect_rate"] == 1.0


def test_pixel_corruption_accuracy_drop_is_measured_against_clean_embedding(world):
    _, _, drop = make_batches(world, "occlusion", 5, 1.0, 300, np.random.default_rng(0))
    assert drop > 0.05
