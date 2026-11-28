"""Conditional generative estimators built on ``zuko`` and ``torchcfm``.

* ``NSF1D``         zuko neural spline flow for p(y | c): exact log-density and sampling.
* ``CFM1D``         conditional (rectified) flow matching for p(y | c) with torchcfm,
                    optional reflow rounds to straighten the paths (few-step sampling).
* ``AmortCFM``      ONE masked flow-matching network giving p(x_j | x_-j) for every j.
* ``JointOTCFM``    joint OT-CFM transport map between two batches.
"""
from __future__ import annotations

import copy

import numpy as np
import torch
import torch.nn as nn
import zuko
from torchcfm.conditional_flow_matching import (ConditionalFlowMatcher,
                                                ExactOptimalTransportConditionalFlowMatcher)


def _set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed % (2**32 - 1))


def _mlp(i: int, o: int, hidden: int, layers: int) -> nn.Sequential:
    mods, d = [], i
    for _ in range(layers):
        mods += [nn.Linear(d, hidden), nn.SiLU()]
        d = hidden
    mods.append(nn.Linear(d, o))
    return nn.Sequential(*mods)


class _Std:
    def fit(self, a: np.ndarray):
        a = np.asarray(a, dtype=np.float64)
        self.mu = a.mean(0)
        self.sd = a.std(0) + 1e-8
        return self

    def __call__(self, a):
        return torch.as_tensor((np.asarray(a, dtype=np.float64) - self.mu) / self.sd, dtype=torch.float32)

    def inv(self, a):
        return np.asarray(a) * self.sd + self.mu


class NSF1D:
    """p(y | c) with a zuko neural spline flow (maximum likelihood, early stopping)."""

    def __init__(self, ctx_dim: int, hidden=(64, 64), transforms: int = 3, bins: int = 8,
                 epochs: int = 60, bs: int = 256, lr: float = 2e-3, patience: int = 8, seed: int = 0):
        self.ctx_dim, self.hidden, self.transforms, self.bins = ctx_dim, list(hidden), transforms, bins
        self.epochs, self.bs, self.lr, self.patience, self.seed = epochs, bs, lr, patience, seed

    def fit(self, y, c):
        _set_seed(self.seed)
        self.sy, self.sc = _Std().fit(np.reshape(y, (-1, 1))), _Std().fit(c)
        Y, C = self.sy(np.reshape(y, (-1, 1))), self.sc(c)
        n = len(Y)
        perm = torch.randperm(n)
        nv = max(int(0.1 * n), 1)
        va, tr = perm[:nv], perm[nv:]
        self.flow = zuko.flows.NSF(features=1, context=self.ctx_dim, transforms=self.transforms,
                                   bins=self.bins, hidden_features=self.hidden)
        opt = torch.optim.Adam(self.flow.parameters(), lr=self.lr)
        best, best_state, bad = np.inf, None, 0
        for _ in range(self.epochs):
            self.flow.train()
            order = tr[torch.randperm(len(tr))]
            for k in range(0, len(order), self.bs):
                b = order[k:k + self.bs]
                loss = -self.flow(C[b]).log_prob(Y[b]).mean()
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.flow.parameters(), 5.0)
                opt.step()
            self.flow.eval()
            with torch.no_grad():
                v = -self.flow(C[va]).log_prob(Y[va]).mean().item()
            if v < best - 1e-4:
                best, best_state, bad = v, copy.deepcopy(self.flow.state_dict()), 0
            else:
                bad += 1
                if bad >= self.patience:
                    break
        if best_state is not None:
            self.flow.load_state_dict(best_state)
        self.flow.eval()
        return self

    @torch.no_grad()
    def sample(self, c, n_rep: int = 1, seed: int | None = None) -> np.ndarray:
        if seed is not None:
            torch.manual_seed(seed)
        out = self.flow(self.sc(c)).sample((n_rep,))[..., 0].numpy()
        return self.sy.inv(out)

    @torch.no_grad()
    def log_prob(self, y, c) -> np.ndarray:
        """Log-density of y in original units (includes the standardisation Jacobian)."""
        lp = self.flow(self.sc(c)).log_prob(self.sy(np.reshape(y, (-1, 1)))).numpy()
        return lp - np.log(self.sy.sd[0])


class CFM1D:
    """p(y | c) via conditional flow matching; sigma=0 ConditionalFlowMatcher == rectified flow."""

    def __init__(self, ctx_dim: int, hidden: int = 128, layers: int = 3, steps: int = 1500,
                 bs: int = 512, lr: float = 2e-3, reflow: int = 0, n_euler: int = 20, seed: int = 0):
        self.ctx_dim, self.hidden, self.layers, self.steps = ctx_dim, hidden, layers, steps
        self.bs, self.lr, self.reflow, self.n_euler, self.seed = bs, lr, reflow, n_euler, seed
        self.matcher = ConditionalFlowMatcher(sigma=0.0)

    def _v(self, net, t, x, c):
        return net(torch.cat([t.expand(len(x), 1), x, c], 1))

    def _train(self, net, X0, X1, C):
        n = len(X1)
        opt = torch.optim.Adam(net.parameters(), lr=self.lr)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, self.steps)
        for _ in range(self.steps):
            b = torch.randint(0, n, (self.bs,))
            t, xt, ut = self.matcher.sample_location_and_conditional_flow(X0[b], X1[b])
            loss = ((self._v(net, t.view(-1, 1), xt, C[b]) - ut) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
        return net

    @torch.no_grad()
    def _ode(self, net, x0, C, n_steps, solver="euler"):
        x, dt = x0.clone(), 1.0 / n_steps
        for k in range(n_steps):
            t = torch.full((1, 1), k * dt)
            if solver == "midpoint":
                xm = x + 0.5 * dt * self._v(net, t, x, C)
                x = x + dt * self._v(net, t + 0.5 * dt, xm, C)
            else:
                x = x + dt * self._v(net, t, x, C)
        return x

    def fit(self, y, c):
        _set_seed(self.seed)
        self.sy, self.sc = _Std().fit(np.reshape(y, (-1, 1))), _Std().fit(c)
        Y, C = self.sy(np.reshape(y, (-1, 1))), self.sc(c)
        X0 = torch.randn_like(Y)
        self.net = self._train(_mlp(2 + self.ctx_dim, 1, self.hidden, self.layers), X0, Y, C)
        for _ in range(self.reflow):
            X0 = torch.randn_like(Y)
            X1 = self._ode(self.net, X0, C, 50, "midpoint")
            net2 = copy.deepcopy(self.net)
            self.net = self._train(net2, X0, X1, C)
        return self

    def sample(self, c, n_rep: int = 1, n_steps: int | None = None, seed: int | None = None,
               z: np.ndarray | None = None, solver: str = "euler") -> np.ndarray:
        if seed is not None:
            torch.manual_seed(seed)
        C = self.sc(c)
        outs = []
        for r in range(n_rep):
            x0 = (torch.randn(len(C), 1) if z is None else torch.as_tensor(z[r], dtype=torch.float32).view(-1, 1))
            outs.append(self._ode(self.net, x0, C, n_steps or self.n_euler, solver)[:, 0].numpy())
        return self.sy.inv(np.stack(outs))


class AmortCFM:
    """Single masked flow-matching network for p(x_j | x_-j), all j at once."""

    def __init__(self, p: int, hidden: int = 256, layers: int = 3, steps: int = 6000, bs: int = 512,
                 lr: float = 1.5e-3, n_euler: int = 20, seed: int = 0):
        self.p, self.hidden, self.layers, self.steps, self.bs = p, hidden, layers, steps, bs
        self.lr, self.n_euler, self.seed = lr, n_euler, seed
        self.matcher = ConditionalFlowMatcher(sigma=0.0)

    def _v(self, t, y, ctx, onehot):
        return self.net(torch.cat([t, y, ctx, onehot], 1))

    def fit(self, X):
        _set_seed(self.seed)
        self.s = _Std().fit(X)
        Z = self.s(X)
        n, p = Z.shape
        self.net = _mlp(2 + 2 * p, 1, self.hidden, self.layers)
        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, self.steps)
        for _ in range(self.steps):
            idx = torch.randint(0, n, (self.bs,))
            j = torch.randint(0, p, (self.bs,))
            onehot = torch.nn.functional.one_hot(j, p).float()
            Zb = Z[idx]
            y1 = (Zb * onehot).sum(1, keepdim=True)
            ctx = Zb * (1 - onehot)
            t, yt, ut = self.matcher.sample_location_and_conditional_flow(torch.randn_like(y1), y1)
            loss = ((self._v(t.view(-1, 1), yt, ctx, onehot) - ut) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
        return self

    @torch.no_grad()
    def sample(self, Xq, j: int, n_rep: int = 1, n_steps: int | None = None, seed: int | None = None):
        if seed is not None:
            torch.manual_seed(seed)
        Z = self.s(Xq)
        onehot = torch.zeros(len(Z), self.p)
        onehot[:, j] = 1.0
        ctx = Z * (1 - onehot)
        K = n_steps or self.n_euler
        outs = []
        for _ in range(n_rep):
            y, dt = torch.randn(len(Z), 1), 1.0 / K
            for k in range(K):
                y = y + dt * self._v(torch.full((len(Z), 1), k * dt), y, ctx, onehot)
            outs.append(y[:, 0].numpy() * self.s.sd[j] + self.s.mu[j])
        return np.stack(outs)


class JointOTCFM:
    """Joint OT-CFM velocity field v(t, x) transporting batch A to batch B."""

    def __init__(self, p: int, hidden: int = 256, layers: int = 3, steps: int = 4000, bs: int = 256,
                 lr: float = 1.5e-3, seed: int = 0):
        self.p, self.hidden, self.layers, self.steps, self.bs, self.lr, self.seed = p, hidden, layers, steps, bs, lr, seed
        self.matcher = ExactOptimalTransportConditionalFlowMatcher(sigma=0.0)

    def fit(self, XA, XB):
        _set_seed(self.seed)
        self.s = _Std().fit(np.vstack([XA, XB]))
        A, B = self.s(XA), self.s(XB)
        self.net = _mlp(1 + self.p, self.p, self.hidden, self.layers)
        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, self.steps)
        for _ in range(self.steps):
            ia, ib = torch.randint(0, len(A), (self.bs,)), torch.randint(0, len(B), (self.bs,))
            t, xt, ut = self.matcher.sample_location_and_conditional_flow(A[ia], B[ib])
            loss = ((self.net(torch.cat([t.view(-1, 1), xt], 1)) - ut) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
        return self

    @torch.no_grad()
    def transport(self, XA, n_steps: int = 40):
        x, dt = self.s(XA), 1.0 / n_steps
        for k in range(n_steps):
            x = x + dt * self.net(torch.cat([torch.full((len(x), 1), k * dt), x], 1))
        return x.numpy()  # standardised coordinates

    def standardise(self, X):
        return self.s(X).numpy()
