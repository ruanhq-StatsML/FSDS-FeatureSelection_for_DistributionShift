"""Monte Carlo check of the design effect of a pooled slope in a panel with time-varying slopes.

y_it = b_t x_it + e_it, x_it ~ AR(1) over t with cross-sectional N(0, 1) margins, b_t = b + tau * eps_t, e_it ~ N(0, 1).
Reports the true s.e. of the pooled OLS slope, the naive i.i.d. s.e., the design effect and the effective sample size.
The theoretical design effect is 1 + N tau^2 / sigma^2.
"""
import numpy as np

rng = np.random.default_rng(0)


def sim(N, T, tau, phi, reps, sig=1.0, b=0.3):
    est, naive = [], []
    for _ in range(reps):
        x = np.zeros((T, N))
        x[0] = rng.standard_normal(N)
        for t in range(1, T):
            x[t] = phi * x[t - 1] + np.sqrt(1 - phi ** 2) * rng.standard_normal(N)
        bt = b + tau * rng.standard_normal(T)
        y = bt[:, None] * x + sig * rng.standard_normal((T, N))
        xf, yf = x.ravel(), y.ravel()
        bh = (xf @ yf) / (xf @ xf)
        est.append(bh)
        r = yf - bh * xf
        naive.append(np.sqrt((r @ r / len(xf)) / (xf @ xf)))
    return np.std(est), np.mean(naive)


print("N T tau phi true_se naive_se deff theory n_eff")
for N, T, tau, phi in [(500, 1, 0.0, 0.0), (500, 12, 0.0, 0.0), (500, 12, 0.1, 0.0), (500, 12, 0.3, 0.0),
                       (2000, 12, 0.1, 0.0), (500, 60, 0.1, 0.0), (500, 60, 0.3, 0.0),
                       (500, 12, 0.1, 0.9), (500, 60, 0.1, 0.9)]:
    se, nv = sim(N, T, tau, phi, reps=800 if T > 20 or phi > 0 else 1500)
    d = (se / nv) ** 2
    print(N, T, tau, phi, round(se, 4), round(nv, 4), round(d, 1), round(1 + N * tau ** 2, 1), int(N * T / d))
