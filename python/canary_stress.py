"""Canary stress test for distribution-shift monitors.

Planted drift is injected into a fraction of shadow traffic and every monitor
is scored on the same cells: detection rate (false-alarm rate in the null
cell), localisation hit rate (the flagged modality is the corrupted one) and
the accuracy the downstream model loses on the perturbed rows. Putting
detection next to harm is the point: a monitor that fires on harmless
brightness changes but misses an attack that halves accuracy is miscalibrated
for the platform even if its type I error is fine.

Dependencies are numpy / scipy / scikit-learn only. The corruptions are numpy
versions of the albumentations families used in imageShiftLocalization.py,
and the attack is the L2-projected gradient step of adversarial_conceptdrift.py
applied in embedding space against a multinomial logistic model, whose input
gradient is analytic.

Run from the python/ directory:

    python canary_stress.py --reps 10 --out canary_results.json
"""
import argparse
import json
import time
from dataclasses import dataclass
from typing import Callable, Dict

import numpy as np
from scipy import ndimage, stats
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from ab_diagnostics import modality_shift_decomposition, split_balance_c2st

ALPHA = 0.05


def _clip(x):
    return np.clip(x, 0.0, 1.0)


def _per_image(fn):
    def apply(imgs, s, rng):
        return np.stack([fn(im, s, rng) for im in imgs])
    return apply


def gaussian_noise(imgs, s, rng):
    return _clip(imgs + rng.normal(scale=[0.05, 0.1, 0.2, 0.3, 0.4][s - 1], size=imgs.shape))


def impulse_noise(imgs, s, rng):
    out = imgs.copy()
    hit = rng.random(imgs.shape) < [0.02, 0.05, 0.1, 0.2, 0.3][s - 1]
    out[hit] = rng.integers(0, 2, size=hit.sum()).astype(float)
    return out


@_per_image
def gaussian_blur(im, s, rng):
    return ndimage.gaussian_filter(im, sigma=[0.4, 0.6, 0.8, 1.0, 1.5][s - 1])


def brightness(imgs, s, rng):
    return _clip(imgs + [0.05, 0.1, 0.2, 0.3, 0.4][s - 1])


def contrast(imgs, s, rng):
    c = [0.8, 0.6, 0.4, 0.3, 0.2][s - 1]
    mean = imgs.mean(axis=(1, 2), keepdims=True)
    return _clip(mean + c * (imgs - mean))


@_per_image
def pixelate(im, s, rng):
    h, w = im.shape
    coarse = im.reshape(h // 2, 2, w // 2, 2).mean(axis=(1, 3)).repeat(2, 0).repeat(2, 1)
    a = [0.2, 0.4, 0.6, 0.8, 1.0][s - 1]
    return (1 - a) * im + a * coarse


@_per_image
def occlusion(im, s, rng):
    k = [1, 2, 3, 4, 5][s - 1]
    out = im.copy()
    r, c = rng.integers(0, im.shape[0] - k + 1), rng.integers(0, im.shape[1] - k + 1)
    out[r:r + k, c:c + k] = 0.0
    return out


@_per_image
def translate(im, s, rng):
    d = [0.5, 1.0, 1.5, 2.0, 2.5][s - 1]
    theta = rng.uniform(0, 2 * np.pi)
    return _clip(ndimage.shift(im, (d * np.sin(theta), d * np.cos(theta)), order=1, mode="constant"))


@_per_image
def rotate(im, s, rng):
    deg = [5, 10, 15, 25, 35][s - 1] * rng.choice([-1, 1])
    return _clip(ndimage.rotate(im, deg, reshape=False, order=1, mode="constant"))


PIXEL_CORRUPTIONS: Dict[str, Callable] = {
    "gaussian_noise": gaussian_noise,
    "impulse_noise": impulse_noise,
    "gaussian_blur": gaussian_blur,
    "brightness": brightness,
    "contrast": contrast,
    "pixelate": pixelate,
    "occlusion": occlusion,
    "translate": translate,
    "rotate": rotate,
}
EMBEDDING_PERTURBATIONS = ("fgsm_l2", "random_l2")
EPS_FRACTION = [0.05, 0.1, 0.2, 0.3, 0.5]


@dataclass
class ImageWorld:
    """Fixed featurizer and downstream model, plus disjoint reference / shadow pools."""

    pca: PCA
    scaler: StandardScaler
    clf: LogisticRegression
    ref_imgs: np.ndarray
    ref_y: np.ndarray
    shadow_imgs: np.ndarray
    shadow_y: np.ndarray
    z_norm: float

    def embed(self, imgs):
        return self.scaler.transform(self.pca.transform(imgs.reshape(len(imgs), -1)))


def build_world(n_components=16, seed=0):
    digits = load_digits()
    imgs, y = digits.images / 16.0, digits.target
    idx = np.random.default_rng(seed).permutation(len(y))
    fit, ref, shadow = np.array_split(idx, 3)
    pca = PCA(n_components=n_components, random_state=seed).fit(imgs[fit].reshape(len(fit), -1))
    scaler = StandardScaler().fit(pca.transform(imgs[fit].reshape(len(fit), -1)))
    z_fit = scaler.transform(pca.transform(imgs[fit].reshape(len(fit), -1)))
    clf = LogisticRegression(max_iter=2000).fit(z_fit, y[fit])
    return ImageWorld(pca, scaler, clf, imgs[ref], y[ref], imgs[shadow], y[shadow],
                      float(np.median(np.linalg.norm(z_fit, axis=1))))


def _l2_step(Z, direction, eps):
    norm = np.linalg.norm(direction, axis=1, keepdims=True) + 1e-12
    return Z + eps * direction / norm


def fgsm_l2(world, Z, y, eps):
    """One L2-normalised gradient-ascent step on the cross-entropy of the downstream model."""
    P = world.clf.predict_proba(Z)
    onehot = (world.clf.classes_[None, :] == y[:, None]).astype(float)
    grad = (P - onehot) @ world.clf.coef_
    return _l2_step(Z, grad, eps)


def random_l2(Z, eps, rng):
    return _l2_step(Z, rng.normal(size=Z.shape), eps)


BLOCKS = {"image": list(range(16)), "text": list(range(16, 24)), "audio": list(range(24, 32))}


def _other_modalities(rng, n):
    return rng.normal(size=(n, 16))


def make_batches(world, perturbation, severity, fraction, n, rng):
    """Reference and shadow batches of n rows each; `fraction` of shadow rows perturbed."""
    ri = rng.choice(len(world.ref_y), size=n, replace=False)
    si = rng.choice(len(world.shadow_y), size=n, replace=False)
    imgs, y = world.shadow_imgs[si].copy(), world.shadow_y[si]
    Z_clean = world.embed(imgs)
    hit = rng.random(n) < fraction if perturbation != "none" else np.zeros(n, bool)
    if perturbation in PIXEL_CORRUPTIONS and hit.any():
        imgs[hit] = PIXEL_CORRUPTIONS[perturbation](imgs[hit], severity, rng)
    Z = world.embed(imgs)
    if perturbation in EMBEDDING_PERTURBATIONS and hit.any():
        eps = EPS_FRACTION[severity - 1] * world.z_norm
        Z[hit] = (fgsm_l2(world, Z[hit], y[hit], eps) if perturbation == "fgsm_l2"
                  else random_l2(Z[hit], eps, rng))
    ref = np.hstack([world.embed(world.ref_imgs[ri]), _other_modalities(rng, n)])
    new = np.hstack([Z, _other_modalities(rng, n)])
    if hit.any():
        acc_drop = float(np.mean(world.clf.predict(Z_clean[hit]) == y[hit])
                         - np.mean(world.clf.predict(Z[hit]) == y[hit]))
    else:
        acc_drop = 0.0
    return ref, new, acc_drop


MONITORS = ("c2st", "mmd_bonferroni", "output_confidence_ks")


def run_monitors(world, ref, new, seed=0, n_perm=200):
    """Input monitors: C2ST (block importance localises) and per-modality MMD with Bonferroni.

    Output monitor: two-sample KS test on the downstream model's top-class
    probability. It only sees the image model, so it cannot localise.
    """
    c2 = split_balance_c2st(ref, new, blocks=BLOCKS, n_perm=n_perm, n_vimp_reps=3, seed=seed)
    mmd = modality_shift_decomposition(ref, new, BLOCKS, max_n=len(ref), n_perm=n_perm, seed=seed)
    p_bonf = {j: min(1.0, p * len(BLOCKS)) for j, p in mmd.p_lomi.items()}
    img = BLOCKS["image"]
    conf_ref = world.clf.predict_proba(ref[:, img]).max(axis=1)
    conf_new = world.clf.predict_proba(new[:, img]).max(axis=1)
    return {
        "c2st": {"detect": c2.p_value < ALPHA, "top": c2.top(1)[0][0]},
        "mmd_bonferroni": {"detect": min(p_bonf.values()) < ALPHA, "top": min(p_bonf, key=p_bonf.get)},
        "output_confidence_ks": {"detect": stats.ks_2samp(conf_ref, conf_new).pvalue < ALPHA, "top": "image"},
    }


def run_cell(world, perturbation, severity, fraction, n, reps, seed):
    rows = []
    for r in range(reps):
        rng = np.random.default_rng(seed + r)
        ref, new, acc_drop = make_batches(world, perturbation, severity, fraction, n, rng)
        rows.append((run_monitors(world, ref, new, seed=r), acc_drop))
    out = {"perturbation": perturbation, "severity": severity, "fraction": fraction, "n": n, "reps": reps,
           "accuracy_drop_on_perturbed": float(np.mean([a for _, a in rows]))}
    for m in MONITORS:
        det = [res[m]["detect"] for res, _ in rows]
        loc = [res[m]["top"] == "image" for res, _ in rows if res[m]["detect"]]
        out[m] = {"detect_rate": float(np.mean(det)),
                  "localised_given_detect": float(np.mean(loc)) if loc and m != "output_confidence_ks" else None}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--null-reps", type=int, default=100)
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--severities", type=int, nargs="*", default=[1, 3])
    ap.add_argument("--fractions", type=float, nargs="*", default=[0.1, 0.3])
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="canary_results.json")
    args = ap.parse_args()

    world = build_world(seed=args.seed)
    names = list(PIXEL_CORRUPTIONS) + list(EMBEDDING_PERTURBATIONS)
    if args.only:
        names = [p for p in names if p in args.only]
    cells = [("none", 1, 0.0)] + [(p, s, f) for p in names for s in args.severities for f in args.fractions]
    results = []
    for k, (p, s, f) in enumerate(cells):
        t0 = time.time()
        reps = args.null_reps if p == "none" else args.reps
        res = run_cell(world, p, s, f, args.n, reps, args.seed + 1000 * k)
        results.append(res)
        print(f"{p:<15} sev={s} frac={f:.1f} acc_drop={res['accuracy_drop_on_perturbed']:+.3f} "
              f"c2st={res['c2st']['detect_rate']:.2f} (loc {res['c2st']['localised_given_detect']}) "
              f"mmd={res['mmd_bonferroni']['detect_rate']:.2f} (loc {res['mmd_bonferroni']['localised_given_detect']}) "
              f"out_ks={res['output_confidence_ks']['detect_rate']:.2f} "
              f"{time.time() - t0:.0f}s", flush=True)
    with open(args.out, "w") as f:
        json.dump({"n": args.n, "reps": args.reps, "cells": results}, f, indent=2)


if __name__ == "__main__":
    main()
