"""Adversarial concept-drift perturbations (wrapper around adversarial_conceptdrift)."""
from adversarial_conceptdrift import (
    AdvConceptDrift,
    adv_concept_drift_pert,
    _train_domain_regressor as train_domain_regressor,
)

def impose_adv_shift(
    df1_X,
    df1_Y,
    df2_X,
    df2_Y,
    method="miFGSM",
    task="reg",
    steps=5,
    eps=0.1,
    n_layers=2,
    n_shift_prop=0.3,
    n_epochs=5,
    device="cpu",
):
    method_map = {
        "miFGSM": "miFGSM",
        "sini_FGSM": "sini_FGSM",
        "vmi_FGSM": "vmi_FGSM",
        "rFGSM": "rFGSM",
        "jitter": "jitter",
        "CD": "CD",
        "forward_CD": "CD",
    }
    if method not in method_map:
        raise ValueError(f"unknown method: {method!r}")
    attack = method_map[method]
    if attack == "CD":
        import numpy as np
        import torch
        from sklearn.preprocessing import StandardScaler
        X = np.vstack([df1_X, df2_X])
        Y = np.hstack([df1_Y, df2_Y])
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        n1 = df1_X.shape[0]
        p = X.shape[1]
        n_shift = max(1, int(np.round(p * n_shift_prop)))
        feature_ind = sorted(np.random.choice(np.arange(p), n_shift, replace=False).tolist())
        model = train_domain_regressor(X_scaled, Y, n_epoch=n_epochs, device=device)
        attacker = AdvConceptDrift(
            model=model,
            attack_feature_list=feature_ind,
            steps=steps,
            eps=eps,
            scaler=False,
            device=device,
        )
        X_adv_t, Y_t = attacker.forward_CD(X_scaled, Y, n1, task=task)
        return X_adv_t.cpu().numpy(), Y_t.cpu().numpy().ravel(), feature_ind, n1
    return adv_concept_drift_pert(
        df1_X,
        df1_Y,
        df2_X,
        df2_Y,
        attack_method=attack,
        n_shift_prop=n_shift_prop,
        n_epoch=n_epochs,
        steps=steps,
        eps=eps,
        device=device,
    )
