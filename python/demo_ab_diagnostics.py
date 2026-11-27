"""Planted-scenario demo for ab_diagnostics.

Run from the python/ directory:  python demo_ab_diagnostics.py
"""
import numpy as np

from ab_diagnostics import aipw_ate, block_loco_mmd, hte_drivers, split_balance_c2st, stage_shift_report


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


if __name__ == "__main__":
    main()
