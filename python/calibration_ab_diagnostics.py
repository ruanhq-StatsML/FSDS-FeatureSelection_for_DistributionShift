"""Monte Carlo calibration of the ab_diagnostics tests.

Reports type I error, power and CI coverage with Wilson 95% intervals.
Run from the python/ directory:

    python calibration_ab_diagnostics.py --scale 1.0 --out calibration_results.json

``--scale`` multiplies every replication count (use 0.2 for a quick check).
"""
import argparse
import json
import time

import numpy as np
from scipy import stats

from ab_diagnostics import (
    aipw_ate,
    dependence_shift_test,
    hte_drivers,
    modality_hte_decomposition,
    modality_info_decomposition,
    split_balance_c2st,
)
from demo_ab_diagnostics import MODALITY_BLOCKS, simulate_modal_hte, simulate_modalities

ALPHA = 0.05


def wilson(k, n, level=0.95):
    if n == 0:
        return (float("nan"), float("nan"))
    z = stats.norm.ppf(0.5 + level / 2)
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return (float(centre - half), float(centre + half))


def summarise(name, hits, reps, extra=None):
    k = int(np.sum(hits))
    lo, hi = wilson(k, reps)
    row = {"check": name, "reps": reps, "rate": k / reps, "ci_low": lo, "ci_high": hi}
    row.update(extra or {})
    print(f"{name:<55} {k:>4}/{reps:<4} = {k / reps:.3f}  [{lo:.3f}, {hi:.3f}]", flush=True)
    return row


def ab_data(rng, n, ate=0.0, hte=0.0, drop=0.0):
    X = rng.normal(size=(n, 6))
    W = rng.binomial(1, 0.5, size=n)
    if drop > 0:
        keep = ~((W == 1) & (X[:, 2] < -0.5) & (rng.random(n) < drop))
        X, W = X[keep], W[keep]
    tau = ate + hte * (np.where(X[:, 3] > 0, 0.5, -0.5))
    Y = 2.0 * X[:, 0] + np.sin(2 * X[:, 1]) + tau * W + rng.normal(size=len(W))
    return X, Y, W


def calib_c2st(reps_null, reps_power, seed):
    rows = []
    for label, drop, reps in [("null", 0.0, reps_null), ("drop 20%", 0.2, reps_power), ("drop 40%", 0.4, reps_power)]:
        hits, top_ok = [], []
        for r in range(reps):
            rng = np.random.default_rng(seed + r)
            X, _, W = ab_data(rng, 2000, drop=drop)
            rep = split_balance_c2st(X[W == 0], X[W == 1], n_perm=200, n_vimp_reps=3, seed=r)
            hits.append(rep.p_value < ALPHA)
            top_ok.append(rep.top(1)[0][0] == "x2")
        extra = {"top_is_x2_given_reject": float(np.mean([t for h, t in zip(hits, top_ok) if h])) if any(hits) else None}
        rows.append(summarise(f"split_balance_c2st reject ({label}, n=2000)", hits, reps, extra))
    return rows


def calib_aipw(reps, seed):
    rows = []
    for ate in (0.0, 0.1):
        cover, reject, ratio = [], [], []
        for r in range(reps):
            rng = np.random.default_rng(seed + 10_000 + r)
            X, Y, W = ab_data(rng, 2000, ate=ate)
            res = aipw_ate(X, Y, W, propensity=0.5, n_folds=3, seed=r)
            cover.append(res.ci[0] <= ate <= res.ci[1])
            reject.append(res.p_value < ALPHA)
            ratio.append(res.variance_ratio_vs_dim)
        rows.append(summarise(f"aipw_ate 95% CI coverage (ATE={ate}, n=2000)", cover, reps,
                              {"median_variance_ratio_vs_dim": float(np.median(ratio))}))
        rows.append(summarise(f"aipw_ate reject H0: ATE=0 (ATE={ate}, n=2000)", reject, reps))
    return rows


def calib_hte(reps_null, reps_power, seed):
    rows = []
    for hte, reps in [(0.0, reps_null), (0.3, reps_power), (0.6, reps_power)]:
        hits, top_ok = [], []
        for r in range(reps):
            rng = np.random.default_rng(seed + 20_000 + r)
            X, Y, W = ab_data(rng, 3000, ate=0.1, hte=hte)
            psi = aipw_ate(X, Y, W, propensity=0.5, n_folds=3, seed=r).pseudo_outcome
            rep = hte_drivers(X, psi, n_perm=3, seed=r)
            hits.append(rep.heterogeneity_p_value < ALPHA)
            top_ok.append(rep.top(1)[0][0] == "x3")
        rows.append(summarise(f"hte_drivers reject (effect gap {hte}, n=3000)", hits, reps,
                              {"top_is_x3": float(np.mean(top_ok))}))
    return rows


def _rho_shift(rng, n, rho_ref, rho_new, mean_shift=0.0, d=4):
    def batch(rho, shift):
        t = rng.normal(size=(n, d)) + shift
        i = rho * (t - shift) + np.sqrt(1 - rho ** 2) * rng.normal(size=(n, d))
        return np.hstack([t, i])
    return batch(rho_ref, 0.0), batch(rho_new, mean_shift)


def calib_dependence(reps_null, reps_power, seed):
    blocks = {"text": [0, 1, 2, 3], "image": [4, 5, 6, 7]}
    rows = []
    cases = [
        ("null", 0.6, 0.6, 0.0, reps_null),
        ("text mean shift 0.6, dependence unchanged", 0.6, 0.6, 0.6, reps_null),
        ("rho 0.6 -> 0.3", 0.6, 0.3, 0.0, reps_power),
        ("rho 0.6 -> 0.0", 0.6, 0.0, 0.0, reps_power),
    ]
    for label, a, b, shift, reps in cases:
        hits = []
        for r in range(reps):
            rng = np.random.default_rng(seed + 30_000 + r)
            ref, new = _rho_shift(rng, 400, a, b, shift)
            hits.append(dependence_shift_test(ref, new, blocks, max_n=400, n_perm=200, seed=r)["p_value"] < ALPHA)
        rows.append(summarise(f"dependence_shift_test reject ({label}, n=400)", hits, reps))
    return rows


def calib_modality_labels(reps, seed):
    rows = []
    for scenario, expected in [
        ("none", {"text": "none", "image": "none", "audio": "none"}),
        ("unique_text", {"text": "unique", "image": "none", "audio": "none"}),
    ]:
        wrong_unshifted, all_correct = [], []
        for r in range(reps):
            ref, new = simulate_modalities(scenario, n=500, seed=seed + 40_000 + r)
            res = modality_info_decomposition(ref, new, MODALITY_BLOCKS, n_repeats=2, seed=r)
            labels = {j: res.pattern(j) for j in MODALITY_BLOCKS}
            wrong_unshifted += [labels[j] != "none" for j, e in expected.items() if e == "none"]
            all_correct.append(labels == expected)
        rows.append(summarise(f"modality pattern: unshifted modality mislabelled ({scenario})",
                              wrong_unshifted, len(wrong_unshifted)))
        rows.append(summarise(f"modality pattern: all labels correct ({scenario})", all_correct, reps))
    return rows


MODAL_HTE_TRUTH = {
    "none": {"text": "none", "image": "none", "audio": "none"},
    "unique_text": {"text": "unique", "image": "none", "audio": "none"},
    "bystander": {"text": "unique", "image": "redundant", "audio": "none"},
    "redundant": {"text": "redundant", "image": "redundant", "audio": "none"},
    "synergy": {"text": "synergistic", "image": "synergistic", "audio": "none"},
}
# Modalities with E[tau | X] = E[tau | X_{-m}] exactly; their raw LOMO test must hold its level.
# In "redundant" the noisy copies carry a negligible but nonzero LOMO, so they are excluded.
MODAL_HTE_LOMO_NULL = {
    "none": ["text", "image", "audio"],
    "unique_text": ["image", "audio"],
    "bystander": ["image", "audio"],
    "redundant": ["audio"],
    "synergy": ["audio"],
}


def calib_modality_hte(reps, seed):
    rows = []
    lomo_null_hits = []
    for scenario, expected in MODAL_HTE_TRUTH.items():
        all_correct, share = [], []
        for r in range(reps):
            s = seed + 50_000 + r
            X, Y, W, blocks = simulate_modal_hte(scenario, seed=s)
            psi = aipw_ate(X, Y, W, propensity=0.5, n_folds=3, seed=r).pseudo_outcome
            res = modality_hte_decomposition(X, psi, blocks, n_splits=3, seed=r)
            labels = {j: res.pattern(j) for j in blocks}
            all_correct.append(labels == expected)
            lomo_null_hits += [res.lomo[j][1] < ALPHA for j in MODAL_HTE_LOMO_NULL[scenario]]
            share.append({j: res.lomo[j][0] / max(res.joint[0], 1e-12) for j in blocks})
        extra = {"median_lomo_share": {j: float(np.median([sh[j] for sh in share])) for j in expected}}
        rows.append(summarise(f"modality HTE: all labels correct ({scenario}, n=6000)", all_correct, reps, extra))
    rows.append(summarise("modality HTE: raw LOMO test rejects a null modality", lomo_null_hits, len(lomo_null_hits)))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="calibration_results.json")
    ap.add_argument("--only", nargs="*", default=None)
    args = ap.parse_args()

    def n(base):
        return max(5, int(round(base * args.scale)))

    suites = {
        "c2st": lambda: calib_c2st(n(200), n(100), args.seed),
        "aipw": lambda: calib_aipw(n(200), args.seed),
        "hte": lambda: calib_hte(n(150), n(60), args.seed),
        "dependence": lambda: calib_dependence(n(200), n(100), args.seed),
        "modality": lambda: calib_modality_labels(n(40), args.seed),
        "modality_hte": lambda: calib_modality_hte(n(20), args.seed),
    }
    results = {}
    for name, fn in suites.items():
        if args.only and name not in args.only:
            continue
        t0 = time.time()
        results[name] = fn()
        print(f"-- {name} done in {time.time() - t0:.0f}s", flush=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
