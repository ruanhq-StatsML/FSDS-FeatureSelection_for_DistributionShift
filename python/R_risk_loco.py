"""Simulation DGPs for concept drift / covariate shift (linear + nonlinear)."""
import numpy as np

P = 20
N_PER_DOMAIN = 1000
CAUSAL_FEATURES = np.arange(4, dtype=int)


def _ar1_cov(p, rho):
    idx = np.arange(p)
    cov = float(rho) ** np.abs(idx[:, None] - idx[None, :])
    return cov + 1e-6 * np.eye(p)


def _draw_X(n, p, rho, rng):
    return rng.multivariate_normal(np.zeros(p), _ar1_cov(p, rho), size=n)


def _standardize(col):
    col = np.asarray(col, dtype=float).ravel()
    return (col - col.mean()) / (col.std() + 1e-8)


def _stack_domains(X0, X1, Y0, Y1):
    X = np.vstack([X0, X1])
    Y = np.concatenate([Y0, Y1])
    W = np.concatenate(
        [np.zeros(len(X0), dtype=int), np.ones(len(X1), dtype=int)]
    )
    return X, Y, W


def _base_beta(rng, p):
    beta = np.zeros(p, dtype=float)
    beta[CAUSAL_FEATURES] = rng.uniform(0.5, 1.5, size=len(CAUSAL_FEATURES))
    return beta


def generate_concept_drift_dgp(
    delta_beta,
    rho,
    seed=0,
    p=P,
    n_per_domain=N_PER_DOMAIN,
):
    rng = np.random.default_rng(seed)
    beta = _base_beta(rng, p)
    X0 = _draw_X(n_per_domain, p, rho, rng)
    X1 = _draw_X(n_per_domain, p, rho, rng)
    Y0 = X0 @ beta + rng.normal(0, 0.1, n_per_domain)
    beta1 = beta.copy()
    beta1[CAUSAL_FEATURES] += float(delta_beta) * np.linspace(
        0.5, 1.5, len(CAUSAL_FEATURES)
    )
    Y1 = X1 @ beta1 + rng.normal(0, 0.1, n_per_domain)
    return _stack_domains(X0, X1, Y0, Y1)


def generate_covariate_shift_dgp(
    gamma,
    rho,
    seed=0,
    p=P,
    n_per_domain=N_PER_DOMAIN,
):
    rng = np.random.default_rng(seed)
    beta = _base_beta(rng, p)
    X0 = _draw_X(n_per_domain, p, rho, rng)
    X1 = _draw_X(n_per_domain, p, rho, rng)
    shift = np.zeros(p, dtype=float)
    shift[CAUSAL_FEATURES] = float(gamma) * rng.uniform(
        0.5, 1.5, size=len(CAUSAL_FEATURES)
    )
    X1 = X1 + shift
    Y0 = X0 @ beta + rng.normal(0, 0.1, n_per_domain)
    Y1 = X1 @ beta + rng.normal(0, 0.1, n_per_domain)
    return _stack_domains(X0, X1, Y0, Y1)


def _nonlinear_delta(X, rng, scale=1.0):
    delta = np.zeros(X.shape[0], dtype=float)
    active = CAUSAL_FEATURES
    for j in active:
        z = _standardize(X[:, j])
        others = active[active != j]
        other = int(rng.choice(others)) if len(others) else int(j)
        z2 = _standardize(X[:, other])
        delta += scale * (
            rng.uniform(0.5, 1.0) * np.sin(z)
            + rng.uniform(0.25, 0.5) * (z ** 2)
            + rng.uniform(0.2, 0.35) * z * z2
        )
    return delta


def generate_nonlinear_concept_drift_dgp(
    delta_beta,
    rho,
    seed=0,
    p=P,
    n_per_domain=N_PER_DOMAIN,
):
    rng = np.random.default_rng(seed)
    beta = _base_beta(rng, p)
    X0 = _draw_X(n_per_domain, p, rho, rng)
    X1 = _draw_X(n_per_domain, p, rho, rng)
    Y0 = X0 @ beta + _nonlinear_delta(X0, rng, scale=0.5) + rng.normal(
        0, 0.1, n_per_domain
    )
    Y1 = (
        X1 @ beta
        + _nonlinear_delta(X1, rng, scale=0.5 + float(delta_beta))
        + rng.normal(0, 0.1, n_per_domain)
    )
    return _stack_domains(X0, X1, Y0, Y1)


def generate_nonlinear_covariate_shift_dgp(
    gamma,
    rho,
    seed=0,
    p=P,
    n_per_domain=N_PER_DOMAIN,
):
    rng = np.random.default_rng(seed)
    beta = _base_beta(rng, p)
    X0 = _draw_X(n_per_domain, p, rho, rng)
    X1 = _draw_X(n_per_domain, p, rho, rng).copy()
    for j in CAUSAL_FEATURES:
        z = _standardize(X1[:, j])
        X1[:, j] += float(gamma) * (
            rng.uniform(0.5, 1.0) * np.sin(z)
            + rng.uniform(0.25, 0.5) * (z ** 2)
        )
        if rng.uniform() > 0.5:
            X1[:, j] += float(gamma) * 0.25 * (z ** 2)
    Y0 = X0 @ beta + rng.normal(0, 0.1, n_per_domain)
    Y1 = X1 @ beta + rng.normal(0, 0.1, n_per_domain)
    return _stack_domains(X0, X1, Y0, Y1)
