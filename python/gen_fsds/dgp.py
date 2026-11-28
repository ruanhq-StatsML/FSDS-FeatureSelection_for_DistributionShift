"""Structural-equation data-generating processes with known ground truth.

A shift is injected at the level of a *mechanism* P(X_j | X_pa(j)). This makes the
conditional diagnosis target exact:

* ``S``           the features whose own mechanism was shifted.
* ``S ∪ pa(S)``   the features whose *full conditional* P(X_k | X_-k) changes
                  (a factor p(x_c | x_pa(c)) with c in S enters the full conditional
                  of every parent of c). This is the exact null/non-null split for
                  conditional-independence style tests of "T ⟂ X_k | X_-k".
* descendants of S inherit a shifted *marginal* but keep an unchanged full
  conditional -- the case where marginal / LOCO methods are fooled.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

NOISE_KINDS = ("normal", "t5", "laplace", "bimodal", "skew")
NONLIN = ("id", "tanh", "sin", "sq")


def _noise(kind: str, size, rng: np.random.Generator) -> np.ndarray:
    """Unit-variance, zero-mean noise of different shapes."""
    if kind == "normal":
        return rng.standard_normal(size)
    if kind == "t5":
        return rng.standard_t(5, size) / np.sqrt(5.0 / 3.0)
    if kind == "laplace":
        return rng.laplace(0.0, 1.0 / np.sqrt(2.0), size)
    if kind == "bimodal":
        sign = np.where(rng.random(size) < 0.5, -1.0, 1.0)
        return sign * 0.9 + np.sqrt(0.19) * rng.standard_normal(size)
    if kind == "skew":
        return rng.exponential(1.0, size) - 1.0
    raise ValueError(kind)


def _g(name: str, x: np.ndarray) -> np.ndarray:
    if name == "id":
        return x
    if name == "tanh":
        return np.tanh(x)
    if name == "sin":
        return np.sin(x)
    if name == "sq":
        return (np.clip(x, -3, 3) ** 2 - 1.0) / np.sqrt(2.0)
    raise ValueError(name)


@dataclass
class Shift:
    """Mechanism shift applied to ``nodes``: kind in {mean, scale, shape}."""

    nodes: tuple
    kind: str = "mean"
    delta: float = 1.0


@dataclass
class SEM:
    p: int = 20
    seed: int = 0
    max_parents: int = 3
    hetero: float = 0.5
    noise_scale: float = 1.0
    order: np.ndarray = field(init=False)
    parents: list = field(init=False)
    weights: list = field(init=False)
    nonlin: list = field(init=False)
    hvec: list = field(init=False)
    noise_kind: list = field(init=False)

    def __post_init__(self):
        rng = np.random.default_rng(self.seed)
        self.order = rng.permutation(self.p)
        self.parents = [[] for _ in range(self.p)]
        self.weights = [None] * self.p
        self.nonlin = [None] * self.p
        self.hvec = [None] * self.p
        self.noise_kind = [NOISE_KINDS[k] for k in rng.integers(0, len(NOISE_KINDS), self.p)]
        for pos, node in enumerate(self.order):
            earlier = self.order[:pos]
            if pos == 0 or rng.random() < 0.12:
                pa = []
            else:
                k = min(len(earlier), int(rng.integers(1, self.max_parents + 1)))
                pa = sorted(rng.choice(earlier, size=k, replace=False).tolist())
            self.parents[node] = pa
            k = len(pa)
            self.weights[node] = rng.choice([-1, 1], k) * rng.uniform(0.5, 1.2, k)
            self.nonlin[node] = [NONLIN[i] for i in rng.integers(0, len(NONLIN), k)]
            self.hvec[node] = rng.uniform(-1, 1, k)

    def children(self, j: int) -> list:
        return [c for c in range(self.p) if j in self.parents[c]]

    def descendants(self, nodes) -> set:
        out, stack = set(), list(nodes)
        while stack:
            for c in self.children(stack.pop()):
                if c not in out:
                    out.add(c)
                    stack.append(c)
        return out

    def truth(self, S) -> dict:
        """Ground-truth feature sets for a mechanism shift on ``S``."""
        S = set(int(s) for s in S)
        pa = set(q for s in S for q in self.parents[s])
        cond = S | pa
        desc = self.descendants(S) - cond
        return {"strict": sorted(S), "cond": sorted(cond), "desc": sorted(desc)}

    def sample(self, n: int, rng: np.random.Generator, shift: Shift | None = None) -> np.ndarray:
        X = np.zeros((n, self.p))
        shifted = set(shift.nodes) if shift is not None else set()
        for j in self.order:
            pa = self.parents[j]
            if pa:
                m = np.zeros(n)
                for w, nl, q in zip(self.weights[j], self.nonlin[j], pa):
                    m += w * _g(nl, X[:, q])
                s = self.noise_scale * np.exp(self.hetero * np.tanh(X[:, pa] @ self.hvec[j]))
            else:
                m, s = np.zeros(n), np.ones(n)
            kind = self.noise_kind[j]
            if j in shifted:
                if shift.kind == "mean":
                    m = m + shift.delta
                elif shift.kind == "scale":
                    s = s * (1.0 + shift.delta)
                elif shift.kind == "shape":
                    kind = "bimodal" if kind != "bimodal" else "normal"
                else:
                    raise ValueError(shift.kind)
            X[:, j] = m + s * _noise(kind, n, rng)
        return X


def covariate_shift_task(sem: SEM, n: int, seed: int, kind: str = "mean", delta: float = 1.0,
                         n_shift: int = 3):
    """Two batches (exist, new) with a mechanism shift on ``n_shift`` random nodes."""
    rng = np.random.default_rng(seed)
    S = tuple(sorted(rng.choice(sem.p, size=n_shift, replace=False).tolist()))
    X0 = sem.sample(n, rng)
    X1 = sem.sample(n, rng, Shift(S, kind, delta))
    return X0, X1, sem.truth(S)


class ConceptFunction:
    """E[Y|X] for the existing batch, plus a drift confined to ``S``."""

    def __init__(self, p: int, S, seed: int, n_base: int = 6):
        rng = np.random.default_rng(seed)
        self.S = list(S)
        self.base = rng.choice(p, size=n_base, replace=False)
        self.a = rng.choice([-1, 1], n_base) * rng.uniform(0.5, 1.5, n_base)
        self.nl = [NONLIN[i] for i in rng.integers(0, len(NONLIN), n_base)]
        self.pair = rng.choice(p, size=2, replace=False)
        self.hw = rng.choice([-1, 1], len(self.S)) * rng.uniform(0.8, 1.2, len(self.S))

    def f(self, X):
        y = np.zeros(len(X))
        for a, nl, j in zip(self.a, self.nl, self.base):
            y += a * _g(nl, X[:, j])
        return y + 0.5 * X[:, self.pair[0]] * X[:, self.pair[1]]

    def h(self, X):
        out = np.zeros(len(X))
        for w, j in zip(self.hw, self.S):
            out += w * np.tanh(X[:, j])
        if len(self.S) >= 2:
            out += 0.7 * np.tanh(X[:, self.S[0]]) * np.tanh(X[:, self.S[1]])
        return out


def concept_drift_task(sem: SEM, n: int, seed: int, kind: str = "mean", delta: float = 1.5,
                       n_drift: int = 3, cov_delta: float = 0.6, n_cov: int = 3, sigma: float = 0.5):
    """Concept drift confined to S, confounded by an *independent* covariate shift on R.

    kind='mean': E[Y|X] changes by delta*h(X_S).
    kind='var' : Var[Y|X] is multiplied by exp(delta*h(X_S)); the conditional mean is unchanged.
    Truth is the strict set S (Y depends on X_j through the drift iff j in S).
    """
    rng = np.random.default_rng(seed)
    perm = rng.permutation(sem.p)
    S = tuple(sorted(perm[:n_drift].tolist()))
    R = tuple(sorted(perm[n_drift:n_drift + n_cov].tolist()))
    cf = ConceptFunction(sem.p, S, seed + 7)
    X0 = sem.sample(n, rng)
    X1 = sem.sample(n, rng, Shift(R, "mean", cov_delta))
    Y0 = cf.f(X0) + sigma * rng.standard_normal(n)
    if kind == "mean":
        Y1 = cf.f(X1) + delta * cf.h(X1) + sigma * rng.standard_normal(n)
    elif kind == "var":
        Y1 = cf.f(X1) + sigma * np.exp(delta * cf.h(X1)) * rng.standard_normal(n)
    else:
        raise ValueError(kind)
    return X0, Y0, X1, Y1, {"strict": sorted(S), "cond": sorted(S), "desc": [], "cov_nodes": list(R)}
