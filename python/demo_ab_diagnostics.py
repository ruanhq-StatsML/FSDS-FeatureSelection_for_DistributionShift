"""Planted-scenario demo for ab_diagnostics.

Run from the python/ directory:  python demo_ab_diagnostics.py
"""
import numpy as np

from ab_diagnostics import (
    aipw_ate,
    block_loco_mmd,
    dependence_shift_test,
    hte_drivers,
    modality_info_decomposition,
    modality_shift_decomposition,
    split_balance_c2st,
    stage_shift_report,
)

MODALITY_BLOCKS = {"text": [0, 1, 2, 3], "image": [4, 5, 6, 7], "audio": [8, 9, 10, 11]}


def simulate_ab(n=8000, p=8, true_ate=0.05, broken_split=False, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p))
    W = rng.binomial(1, 0.5, size=n)
    if broken_split:
        drop = (W == 1) & (X[:, 2] < -0.5) & (rng.random(n) < 0.4)
        X, W = X[~drop], W[~drop]
    tau = true_ate + 0.3 * (X[:, 3] > 0) - 0.15
    Y = 2.0 * X[:, 0] + np.sin(2 * X[:, 1]) + 0.5 * X[:, 4] ** 2 + tau * W + rng.normal(size=len(W))
    return X, Y, W


def simulate_cross_modal(n=1500, d=4, rho=0.8, seed=0):
    rng = np.random.default_rng(seed)
    def batch(sign):
        text = rng.normal(size=(n, d))
        image = sign * rho * text + np.sqrt(1 - rho ** 2) * rng.normal(size=(n, d))
        return np.hstack([text, image])
    return batch(+1), batch(-1), {"text": list(range(d)), "image": list(range(d, 2 * d))}


def simulate_modalities(scenario, n=800, d=4, shift=0.6, seed=0):
    """text / image / audio blocks; audio never shifts.

    unique_text      text mean shifts, text and image independent
    redundant        a shared latent shifts, so text and image shift together
    dependence_flip  text-image correlation flips sign, marginals unchanged
    """
    rng = np.random.default_rng(seed)

    def batch(new):
        if scenario == "none":
            t = rng.normal(size=(n, d))
            i = 0.8 * t + 0.6 * rng.normal(size=(n, d))
        elif scenario == "unique_text":
            t = rng.normal(size=(n, d)) + (shift if new else 0.0)
            i = rng.normal(size=(n, d))
        elif scenario == "redundant":
            z = rng.normal(size=(n, d)) + (shift if new else 0.0)
            t = z + 0.3 * rng.normal(size=(n, d))
            i = z + 0.3 * rng.normal(size=(n, d))
        elif scenario == "dependence_flip":
            t = rng.normal(size=(n, d))
            i = (-0.8 if new else 0.8) * t + 0.6 * rng.normal(size=(n, d))
        else:
            raise ValueError(scenario)
        return np.hstack([t, i, rng.normal(size=(n, d))])

    return batch(False), batch(True)


def main():
    print("== 1. healthy randomised split")
    X, Y, W = simulate_ab()
    bal = split_balance_c2st(X[W == 0], X[W == 1])
    print(f"C2ST AUC={bal.auc:.3f} p={bal.p_value:.3f}")

    print("\n== 2. ATE: diff-in-means vs CUPED (x0 as pre-period metric) vs cross-fitted AIPW")
    res = aipw_ate(X, Y, W, propensity=0.5, pre_period_col=0)
    print(f"DIM   {res.diff_in_means:+.4f} (se {res.diff_in_means_se:.4f})")
    print(f"CUPED {res.cuped:+.4f} (se {res.cuped_se:.4f})")
    print(f"AIPW  {res.ate:+.4f} (se {res.se:.4f}) CI=({res.ci[0]:+.4f}, {res.ci[1]:+.4f}) p={res.p_value:.4f}")
    print(f"AIPW variance / DIM variance = {res.variance_ratio_vs_dim:.3f}")

    print("\n== 3. HTE drivers on the AIPW pseudo-outcome (true driver: x3)")
    hte = hte_drivers(X, res.pseudo_outcome)
    print(f"heterogeneity gain={hte.heterogeneity_gain:.4f} p={hte.heterogeneity_p_value:.4f}")
    print("top:", [(k, round(v, 4)) for k, v in hte.top(3)])

    print("\n== 4. broken split (treatment drops x2 < -0.5 users)")
    Xb, Yb, Wb = simulate_ab(broken_split=True)
    bal_b = split_balance_c2st(Xb[Wb == 0], Xb[Wb == 1])
    print(f"C2ST AUC={bal_b.auc:.3f} p={bal_b.p_value:.3f} top:", [(k, round(v, 4)) for k, v in bal_b.top(3)])

    print("\n== 5. cross-modal dependence shift (marginals unchanged)")
    ref, new, blocks = simulate_cross_modal()
    for name, v in block_loco_mmd(ref, new, blocks).items():
        print(f"{name:>9}: loco={v['loco']:.4f} marginal={v['marginal']:.4f}")

    print("\n== 6. gray-release stage comparison: concept drift on x3, same traffic mix")
    rng = np.random.default_rng(1)
    X1 = rng.normal(size=(3000, 6))
    X2 = rng.normal(size=(3000, 6))
    Y1 = X1[:, 0] + rng.normal(size=3000)
    Y2 = X2[:, 0] + 0.8 * X2[:, 3] + rng.normal(size=3000)
    rep = stage_shift_report(X1, Y1, X2, Y2)
    print(f"covariate arm: AUC={rep['covariate'].auc:.3f} p={rep['covariate'].p_value:.3f}")
    print(f"concept arm: heterogeneity p={rep['concept'].heterogeneity_p_value:.4f} top:",
          [(k, round(v, 4)) for k, v in rep["concept"].top(3)])

    print("\n== 7. modality decomposition: LOMI / LOMO / Shapley / interaction, plus copula dependence test")
    for scenario in ["none", "unique_text", "redundant", "dependence_flip"]:
        ref, new = simulate_modalities(scenario)
        r = modality_shift_decomposition(ref, new, MODALITY_BLOCKS)
        dep = dependence_shift_test(ref, new, MODALITY_BLOCKS)
        inter = r.interaction[frozenset(["text", "image"])]
        print(f"-- {scenario}: v(M)={r.value[frozenset(MODALITY_BLOCKS)]:.4f} p_joint={r.p_joint:.3f} "
              f"interaction(text,image)={inter:+.4f} dependence p={dep['p_value']:.3f}")
        for j in MODALITY_BLOCKS:
            lo, hi = r.order_range[j]
            print(f"   {j:>5}: LOMI={r.lomi[j]:+.4f} (p={r.p_lomi[j]:.3f}) LOMO={r.lomo[j]:+.4f} "
                  f"Shapley={r.shapley[j]:+.4f} order range [{lo:+.4f}, {hi:+.4f}]")

    print("\n== 8. monotone decomposition: v(S) = I(W; X_S) from cross-fitted domain classifiers")
    for scenario in ["none", "unique_text", "redundant", "dependence_flip"]:
        ref, new = simulate_modalities(scenario)
        r = modality_info_decomposition(ref, new, MODALITY_BLOCKS)
        inter = r.interaction[frozenset(["text", "image"])]
        print(f"-- {scenario}: v(M)={r.value[frozenset(MODALITY_BLOCKS)][0]:+.3f} "
              f"interaction(text,image)={inter[0]:+.3f} (z={inter[0] / inter[1]:+.1f})")
        for j in MODALITY_BLOCKS:
            (mi, mi_se), (mo, mo_se) = r.lomi[j], r.lomo[j]
            print(f"   {j:>5}: LOMI={mi:+.3f} (z={mi / mi_se:+.1f}) LOMO={mo:+.3f} (z={mo / mo_se:+.1f}) "
                  f"Shapley={r.shapley[j][0]:+.3f} -> {r.pattern(j)}")


if __name__ == "__main__":
    main()
