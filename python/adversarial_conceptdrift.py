"""Adversarial concept drift: MI-FGSM / SI-NI / VMI / R-FGSM / jitter on domain 1
   Add a mask on a subset of the features
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler

ATTACK_METHODS = (
    "miFGSM",
    "sini_FGSM",
    "vmi_FGSM",
    "rFGSM",
    "jitter",
)


class DomainRegressor(nn.Module):
    def __init__(self, input_dim, n_layers=2, dropout=0.15):
        super().__init__()
        dims = np.round(
            np.exp(np.linspace(np.log(max(input_dim, 2)), 0, n_layers))
        ).astype(int)
        dims = np.maximum(dims, 1)
        layers_dict = []
        for i in range(n_layers - 1):
            in_dim = dims[i]
            out_dim = dims[i + 1]
            layers_dict.append(nn.Linear(in_dim, out_dim))
            layers_dict.append(nn.BatchNorm1d(out_dim))
            layers_dict.append(nn.ReLU())
            if dropout > 0:
                layers_dict.append(nn.Dropout(dropout))
        layers_dict.append(nn.Linear(dims[-1], 1))
        self.net = nn.Sequential(*layers_dict)

    def forward(self, X):
        return self.net(X).squeeze(-1)


class AdvConceptDrift:
    def __init__(
        self,
        model,
        attack_feature_list=None,
        decay=0.7,
        scaler=True,
        steps=10,
        alpha=0.01,
        eps=0.01,
        noise_mag=0.1,
        N=5,
        m=4,
        random_start=True,
        device="cpu",
    ):
        self.steps = steps
        self.attack_feature_list = attack_feature_list or [0, 1, 2, 3]
        self.random_start = random_start
        self.eps = eps
        self.noise_mag = noise_mag
        self.scaler = scaler
        self.decay = decay
        self.alpha = alpha
        self.N = N
        self.m = m
        self.model = model
        self.device = device

    def _prepare_data(self, df_X, df_Y):
        X_np = np.asarray(df_X, dtype=float)
        Y_np = np.asarray(df_Y, dtype=float).ravel()
        if self.scaler:
            X_scaled_np = StandardScaler().fit_transform(X_np)
        else:
            X_scaled_np = X_np
        mask_np = np.zeros_like(X_scaled_np)
        mask_np[:, self.attack_feature_list] = 1.0
        X_scaled = torch.tensor(X_scaled_np, dtype=torch.float32, device=self.device)
        Y = torch.tensor(Y_np, dtype=torch.float32, device=self.device)
        mask = torch.tensor(mask_np, dtype=torch.float32, device=self.device)
        return X_scaled, Y, mask

    def _project_l2(self, adv_X, X_scaled, eps, mask):
        delta = (adv_X - X_scaled) * mask
        delta_norm = torch.norm(delta, p=2, dim=1, keepdim=True)
        scale = torch.clamp(eps / (delta_norm + 1e-8), max=1.0)
        return X_scaled + delta * scale

    def _loss_fn(self, task):
        if task == "reg":
            return nn.MSELoss(reduction="mean")
        return nn.CrossEntropyLoss()

    def forward_miFGSM(self, df_X, df_Y, n1, task="reg"):
        X_scaled, Y, mask = self._prepare_data(df_X, df_Y)
        adv_X = X_scaled.clone().detach().requires_grad_(True)
        momentum = torch.zeros_like(adv_X).detach()
        loss_fn = self._loss_fn(task)
        for _ in range(self.steps):
            output = self.model(adv_X)
            loss = loss_fn(output, Y)
            grads = torch.autograd.grad(loss, adv_X, retain_graph=False)[0]
            grads = grads * mask
            grad_norm = grads / (
                torch.mean(torch.abs(grads), dim=1, keepdim=True) + 1e-8
            )
            grad_norm = grad_norm + self.decay * momentum
            momentum = grad_norm
            random_factor = 0.15 * torch.rand(1).item() + 0.1
            adv_X = adv_X + random_factor * (
                grad_norm / (grad_norm.norm(p=2, dim=1, keepdim=True) + 1e-8)
            )
            delta = self._project_l2(adv_X, X_scaled, self.eps, mask)
            adv_X = (X_scaled + delta).detach().requires_grad_(True)
            adv_X = adv_X * mask + X_scaled * (1 - mask)
        X_output = torch.cat([adv_X[:n1], X_scaled[n1:]], dim=0)
        return X_output.detach(), Y.detach()

    def sini_FGSM(self, df_X, df_Y, n1, task="reg"):
        X_scaled, Y, mask = self._prepare_data(df_X, df_Y)
        adv_X = X_scaled.clone().detach().requires_grad_(True)
        momentum = torch.zeros_like(adv_X).detach()
        loss_fn = self._loss_fn(task)
        for _ in range(self.steps):
            adv_X = adv_X.detach().requires_grad_(True)
            nes_X = adv_X + self.decay * self.alpha * momentum
            adv_grad = torch.zeros_like(adv_X)
            for i in range(self.m):
                x_i = (nes_X / torch.pow(torch.tensor(2.0, device=self.device), i)).detach()
                x_i.requires_grad_(True)
                output = self.model(x_i)
                cost = loss_fn(output, Y)
                adv_grad = adv_grad + (
                    torch.autograd.grad(cost, x_i, retain_graph=False)[0] * mask
                )
            adv_grad = adv_grad / self.m
            grad = (
                adv_grad / (torch.mean(torch.abs(adv_grad), dim=1, keepdim=True) + 1e-8)
                + self.decay * momentum
            )
            grad = grad * mask
            momentum = grad
            random_factor = 0.15 * torch.rand(1).item() + 0.1
            adv_X = adv_X + random_factor * (
                grad / (grad.norm(p=2, dim=1, keepdim=True) + 1e-8)
            )
            delta = torch.clamp(adv_X - X_scaled, -self.eps, self.eps)
            adv_X = (X_scaled + delta).detach().requires_grad_(True)
            adv_X = adv_X * mask + X_scaled * (1 - mask)
        X_output = torch.cat([adv_X[:n1], X_scaled[n1:]], dim=0)
        return X_output.detach(), Y.detach()

    def vmi_FGSM(self, df_X, df_Y, n1, task="reg"):
        X_scaled, Y, mask = self._prepare_data(df_X, df_Y)
        adv_X = X_scaled.clone().detach().requires_grad_(True)
        loss_fn = self._loss_fn(task)
        V = torch.zeros_like(adv_X)
        momentum = torch.zeros_like(adv_X)
        for _ in range(self.steps):
            adv_X = adv_X.detach().requires_grad_(True)
            outputs = self.model(adv_X)
            cost = loss_fn(outputs, Y)
            adv_grads = (
                torch.autograd.grad(cost, adv_X, retain_graph=False)[0] * mask
            )
            grad = (adv_grads + V) / (
                torch.mean(torch.abs(adv_grads + V), dim=1, keepdim=True) + 1e-8
            )
            grad = grad * mask + momentum * self.decay
            momentum = grad
            GV_grad = torch.zeros_like(adv_X)
            for _ in range(self.N):
                neighborX = (
                    adv_X + torch.rand_like(adv_X).uniform_(-0.005, 0.005)
                ).detach().requires_grad_(True)
                outputs_n = self.model(neighborX)
                cost_n = loss_fn(outputs_n, Y)
                GV_grad = GV_grad + (
                    torch.autograd.grad(cost_n, neighborX, retain_graph=False)[0]
                    * mask
                )
            V = GV_grad / self.N - adv_grads
            random_factor = 0.15 * torch.rand(1).item() + 0.1
            adv_X = adv_X + random_factor * (
                grad / (grad.norm(p=2, dim=1, keepdim=True) + 1e-8)
            )
            delta = torch.clamp(adv_X - X_scaled, -self.eps, self.eps)
            adv_X = (X_scaled + delta).detach().requires_grad_(True)
            adv_X = adv_X * mask + X_scaled * (1 - mask)
        X_output = torch.cat([adv_X[:n1], X_scaled[n1:]], dim=0)
        return X_output.detach(), Y.detach()

    def forward_rFGSM(self, df_X, df_Y, n1, task="reg"):
        X_scaled, Y, mask = self._prepare_data(df_X, df_Y)
        adv_X = X_scaled.clone().detach().requires_grad_(True)
        loss_fn = self._loss_fn(task)
        for _ in range(self.steps):
            adv_X = adv_X.detach().requires_grad_(True)
            output = self.model(adv_X)
            loss = loss_fn(output, Y)
            grads = torch.autograd.grad(loss, adv_X, retain_graph=False)[0] * mask
            random_factor = 0.15 * torch.rand(1).item() + 0.1
            adv_X = adv_X + random_factor * (
                grads / (grads.norm(p=2, dim=1, keepdim=True) + 1e-8)
            )
            delta = self._project_l2(adv_X, X_scaled, self.eps, mask)
            adv_X = (X_scaled + delta).detach().requires_grad_(True)
            adv_X = adv_X * mask + X_scaled * (1 - mask)
        X_output = torch.cat([adv_X[:n1], X_scaled[n1:]], dim=0)
        return X_output.detach(), Y.detach()

    def forward_jitter(self, df_X, df_Y, n1, task="reg"):
        X_scaled, Y, mask = self._prepare_data(df_X, df_Y)
        adv_X = X_scaled.clone().detach().requires_grad_(True)
        if self.random_start:
            noise = torch.empty_like(adv_X).uniform_(-self.eps, self.eps)
            adv_X = (adv_X + noise).clamp(-1.5, 1.5).detach().requires_grad_(True)
        loss_fn = self._loss_fn(task)
        for _ in range(self.steps):
            adv_X = adv_X.detach().requires_grad_(True)
            output = self.model(adv_X)
            loss = loss_fn(output, Y)
            grads = torch.autograd.grad(loss, adv_X, retain_graph=False)[0] * mask
            random_factor = 0.2 * torch.rand(1).item() + 0.1
            adv_X = adv_X + random_factor * (
                grads / (grads.norm(p=2, dim=1, keepdim=True) + 1e-8)
            )
            delta = self._project_l2(adv_X, X_scaled, self.eps, mask)
            adv_X = (X_scaled + delta).detach().requires_grad_(True)
            adv_X = adv_X * mask + X_scaled * (1 - mask)
        X_output = torch.cat([adv_X[:n1], X_scaled[n1:]], dim=0)
        return X_output.detach(), Y.detach()


def _train_domain_regressor(X, Y, n_epoch=4, batch_size=64, device="cpu"):
    p = X.shape[1]
    model = DomainRegressor(input_dim=p, n_layers=2, dropout=0.15).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    X_t = torch.tensor(X, dtype=torch.float32, device=device)
    Y_t = torch.tensor(Y, dtype=torch.float32, device=device)
    for epoch in range(n_epoch):
        model.train()
        perm = torch.randperm(len(X_t), device=device)
        for i in range(0, len(X_t), batch_size):
            idx = perm[i : i + batch_size]
            optimizer.zero_grad()
            loss = criterion(model(X_t[idx]), Y_t[idx])
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            train_loss = criterion(model(X_t), Y_t).item()
        print(f"  epoch {epoch + 1}/{n_epoch} loss={train_loss:.4f}", flush=True)
    return model


def adv_concept_drift_pert(
    df1_X,
    df1_Y,
    df2_X,
    df2_Y,
    attack_method="miFGSM",
    n_shift_prop=0.3,
    n_epoch=4,
    steps=5,
    eps=0.15,
    device="cpu",
):
    """Train regressor, adversarially perturb domain-1 X on selected features."""
    X = np.vstack([df1_X, df2_X])
    Y = np.hstack([df1_Y, df2_Y])
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    n1 = df1_X.shape[0]
    p = X.shape[1]
    n_shift = max(1, int(np.round(p * n_shift_prop)))
    attack_features = np.arange(n_shift, dtype=int)
    feature_ind = attack_features.copy()

    model = _train_domain_regressor(X_scaled, Y, n_epoch=n_epoch, device=device)
    model.eval()

    attacker = AdvConceptDrift(
        model=model,
        attack_feature_list=attack_features.tolist(),
        steps=steps,
        eps=eps,
        device=device,
    )
    method_map = {
        "miFGSM": attacker.forward_miFGSM,
        "sini_FGSM": attacker.sini_FGSM,
        "vmi_FGSM": attacker.vmi_FGSM,
        "rFGSM": attacker.forward_rFGSM,
        "jitter": attacker.forward_jitter,
    }
    if attack_method not in method_map:
        raise ValueError(f"unknown attack_method: {attack_method}")
    X_adv_t, Y_t = method_map[attack_method](X_scaled, Y, n1, task="reg")
    X_adv = X_adv_t.cpu().numpy()
    Y_out = Y_t.cpu().numpy().ravel()
    return X_adv, Y_out, feature_ind, n1
