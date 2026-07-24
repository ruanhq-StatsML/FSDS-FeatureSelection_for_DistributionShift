#List of the synthetic data generating processes for injecting into the datasets,
#Including simulations for covariate shift and concept drift
import numpy as np
import pandas as pd
from scipy.linalg import toeplitz

#By default the shifted features is the first 4 of the 20 features in total.
P = 20
N_PER_DOMAIN = 1000
SHIFTED_FEATURES = np.arange(4, dtype=int)


def banded_toeplitz_corr(rho, p=20, bandwidth=6):
    """AR(1)-style banded correlation: corr[i,j] = rho^|i-j| within bandwidth."""
    from scipy.linalg import toeplitz
    corr = toeplitz(rho ** np.arange(p))
    return np.triu(np.tril(corr, bandwidth), -bandwidth)

"""
Concept drift: same covariate distribution, but CATE changes after n/2.
rho controls feature correlation (nuisance difficulty); delta_beta(Concept Drift Degree) scales
the post-shift treatment effect on the first four features, build on top of the 
multivariate normal data-generating processes.
"""
def generate_concept_drift_dgp(n=2000, p=20, delta_beta=0.0, rho=0.0, seed=None):
    rng = np.random.default_rng(seed)
    corr = banded_toeplitz_corr(float(rho), p=p)
    X0 = rng.multivariate_normal(mean=np.zeros(p), cov=corr, size=n)
    beta0 = np.ones(p)
    Y = X0 @ beta0 + rng.normal(0, 1, n)
    beta = np.zeros(p)
    beta[:4] = np.arange(4) * float(delta_beta)
    Y[n // 2 :] = Y[n // 2 :] + (X0[n // 2 :, :] @ beta)
    W = np.concatenate([np.zeros(n // 2), np.ones(n - n // 2)])
    return X0, Y, W


def generate_covariate_shift_dgp(n=2000, p=20, gamma=0.0, rho=0.0, seed=None):
    """
    Covariate shift: outcome mechanism fixed, but P(X) differs pre/post.
    The first half gets an additive shift in X[:, :4]; gamma controls both
    outcome strength and shift magnitude on those four features(Covariate Shift Degree)
    """
    rng = np.random.default_rng(seed)
    corr = banded_toeplitz_corr(float(rho), p=p)
    X0 = rng.multivariate_normal(mean=np.zeros(p), cov=corr, size=n)
    beta0 = np.zeros(p)
    beta0[:4] = float(gamma)
    shift_score = X0[: n // 2, :4] @ beta0[:4]
    X0[: n // 2, :4] = X0[: n // 2, :4] + np.tile(
        shift_score.reshape(-1, 1), (1, 4)
    )
    Y = X0 @ beta0 + rng.normal(0, 1, n)
    W = np.concatenate([np.zeros(n // 2), np.ones(n - n // 2)])
    return X0, Y, W



def nonlinear_delta(X, rng, scale=1.0):
	'''
	Nonlinear coefficients here:
	'''
    delta = np.zeros(X.shape[0], dtype=float)
    active = SHIFTED_FEATURES
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
    beta[SHIFTED_FEATURES] = rng.uniform(0.5, 1.5, size=len(SHIFTED_FEATURES))
    return beta


def generate_nonlinear_concept_drift_dgp(
    delta_beta,
    rho,
    seed=0,
    p=P,
    n_per_domain=N_PER_DOMAIN,
):
    '''
    Impose the nonlinear concept drift on top of the current data-generating processes:
    '''
    rng = np.random.default_rng(seed)
    beta = _base_beta(rng, p)
    X0 = _draw_X(n_per_domain, p, rho, rng)
    X1 = _draw_X(n_per_domain, p, rho, rng)
    Y0 = X0 @ beta + nonlinear_delta(X0, rng, scale=0.5) + rng.normal(0, 0.1, n_per_domain)
    Y1 = (X1 @ beta + nonlinear_delta(X1, rng, scale=0.5 + float(delta_beta)) + rng.normal(0, 0.1, n_per_domain))
    return _stack_domains(X0, X1, Y0, Y1)


def generate_nonlinear_covariate_shift_dgp(
    gamma,
    rho,
    seed=0,
    p=P,
    n_per_domain=N_PER_DOMAIN,
):
    """
    Nonlinear Covariate shift: outcome mechanism fixed, but P(X) differs pre/post.
    The first half gets an quadratic & sin shift in X[:, :4]; gamma controls both
    outcome strength and shift magnitude on those four features(Covariate Shift Degree)
    """
    rng = np.random.default_rng(seed)
    beta = _base_beta(rng, p)
    X0 = _draw_X(n_per_domain, p, rho, rng)
    X1 = _draw_X(n_per_domain, p, rho, rng).copy()
    for j in SHIFTED_FEATURES:
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




def synthetic_injection_nonlinear_interaction(
    df1_X, df1_Y, df2_X, df2_Y, n_prop=0.35, seed=2026
):
    '''
    Nonlinear Interaction Injection, somewhat messy, 
    including nonlinear term(sin), quadratic term, interaction term
    '''
    rng = np.random.default_rng(seed)
    n2, p = df2_X.shape
    df2_X = np.asarray(df2_X, dtype=float).copy()
    n_shift = max(1, int(np.round(p * n_prop)))
    feature_ind = rng.choice(np.arange(p), size=n_shift, replace=False)
    delta = np.zeros(n2, dtype=float)
    for j in feature_ind:
        X_col = df2_X[:, j]
        X_standardized = (X_col - X_col.mean()) / (X_col.std() + 1e-8)
        c1_nonlinear = rng.uniform(0.5, 1.0)
        c2_quadratic = rng.uniform(0.25, 0.5)
        others = np.setdiff1d(feature_ind, j)
        other_idx = int(j) if len(others) == 0 else int(rng.choice(others))
        X_interact = df2_X[:, other_idx]
        X_interact_standardized = (X_interact - X_interact.mean()) / (
            X_interact.std() + 1e-7
        )
        c3_interact = rng.uniform(0.2, 0.35)
        delta += (
            c1_nonlinear * np.sin(X_standardized)
            + c2_quadratic * (X_standardized ** 2)
            + c3_interact * X_standardized * X_interact_standardized
        )
        if rng.uniform(0, 1) > 0.5:
            df2_X[:, j] = df2_X[:, j] + (X_standardized ** 2) * 0.25

    df2_Y = df2_Y + delta + rng.normal(0, 0.05, n2)
    df1 = np.hstack([df1_X, np.asarray(df1_Y, dtype=float).reshape(-1, 1)])
    df2 = np.hstack([df2_X, np.asarray(df2_Y, dtype=float).reshape(-1, 1)])
    return df1, df2, feature_ind



def synthetic_injection_linear(df1_X, df1_Y, df2_X, df2_Y, n_prop=0.3, seed=0):
    '''
    Linear Interaction Injection:
    '''
    rng = np.random.default_rng(seed)
    n2, p = df2_X.shape
    n_shift = max(1, int(np.round(p * n_prop)))
    feature_ind = rng.choice(p, size=n_shift, replace=False)
    beta_shift = np.zeros(p)
    beta_shift[feature_ind] = np.linspace(0, 1, len(feature_ind))
    df2_Y = df2_Y + df2_X @ beta_shift + rng.normal(0, 0.05, n2)
    df1 = np.hstack([df1_X, np.asarray(df1_Y, dtype=float).reshape(-1, 1)])
    df2 = np.hstack([df2_X, np.asarray(df2_Y, dtype=float).reshape(-1, 1)])
    return df1, df2, feature_ind


def synthetic_injection_nonlinear(
    df1_X, df1_Y, df2_X, df2_Y, n_prop=0.3, seed=0, strength=1.0
):
    """Concept drift via sin and quadratic terms on ~n_prop of features (domain 2)."""
    rng = np.random.default_rng(seed)
    n2, p = df2_X.shape
    n_shift = max(1, int(np.round(p * n_prop)))
    feature_ind = np.sort(rng.choice(p, size=n_shift, replace=False))

    delta = np.zeros(n2, dtype=float)
    for j in feature_ind:
        x = df2_X[:, j]
        x_std = (x - x.mean()) / (x.std() + 1e-8)
        c_sin = rng.uniform(0.3, strength)
        c_quad = rng.uniform(0.3, strength)
        delta += c_sin * np.sin(x_std) + c_quad * (x_std ** 2)

    df2_Y = df2_Y + delta + rng.normal(0, 0.05, n2)
    df1 = np.hstack([df1_X, np.asarray(df1_Y, dtype=float).reshape(-1, 1)])
    df2 = np.hstack([df2_X, np.asarray(df2_Y, dtype=float).reshape(-1, 1)])
    return df1, df2, feature_ind


def synthetic_injection_interaction(
    df1_X, df1_Y, df2_X, df2_Y, n_prop=0.3, seed=0, strength=1.0
):
    """Concept drift via pairwise interactions on ~n_prop of features (domain 2 only)."""
    rng = np.random.default_rng(seed)
    n2, p = df2_X.shape
    n_shift = max(2, int(np.round(p * n_prop)))
    feature_ind = np.sort(rng.choice(p, size=n_shift, replace=False))

    pairs = list(itertools.combinations(feature_ind, 2))
    if not pairs:
        pairs = [(int(feature_ind[0]), int(feature_ind[0]))]

    delta = np.zeros(n2, dtype=float)
    coefs = rng.uniform(0.3, strength, size=len(pairs))
    for (j, k), c in zip(pairs, coefs):
        delta += c * df2_X[:, j] * df2_X[:, k]

    df2_Y = df2_Y + delta + rng.normal(0, 0.05, n2)
    df1 = np.hstack([df1_X, np.asarray(df1_Y, dtype=float).reshape(-1, 1)])
    df2 = np.hstack([df2_X, np.asarray(df2_Y, dtype=float).reshape(-1, 1)])
    return df1, df2, feature_ind











