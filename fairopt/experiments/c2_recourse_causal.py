"""
C2: The Cost of Causal Ignorance — Latent Intervention Effort in OT.
Compares Standard OT vs Causal OT across 4 fairness datasets with K-fold CV.
Key metric: True Intervention Effort (latent L2) vs Apparent Cost (ambient L2).
Uncertainty reported as mean ± std across CV folds.
"""
from __future__ import annotations

import torch
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold

from fairopt.core.cost import pairwise_l2
from fairopt.core.metrics import barycentric_projection
from fairopt.data.datasets import load_dataset

import ot

RESULTS_DIR = Path(__file__).resolve().parents[2] / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
LOG = RESULTS_DIR / "c2_run.log"

SINKHORN_EPS = 0.5
SINKHORN_MAX_ITER = 100000
SINKHORN_TOL = 1e-6
MAX_SAMPLES = 50000
N_FOLDS = 5


def log(msg: str) -> None:
    print(msg, flush=True)
    with open(LOG, "a") as f:
        f.write(msg + "\n")
        f.flush()


def fit_linear_scm(X: torch.Tensor, chain_order):
    p = X.shape[1]
    X_np = X.numpy()
    M = np.eye(p)
    for k in range(1, p):
        parent_idx = chain_order[k - 1]
        child_idx = chain_order[k]
        parent = X_np[:, parent_idx]
        child = X_np[:, child_idx]
        beta = np.linalg.lstsq(parent[:, None], child, rcond=None)[0][0]
        M[child_idx, parent_idx] = -beta
    M_t = torch.from_numpy(M).float()
    M_inv = torch.linalg.inv(M_t)
    M_inv_T = M_inv.T
    return M_t, M_inv_T


def compute_anisotropy(M: torch.Tensor) -> float:
    G = M.T @ M
    evals = torch.linalg.eigvalsh(G)
    return (evals.max() / evals.min()).item()

def compute_transport(source: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    n_src = source.shape[0]
    n_tgt = target.shape[0]

    mu = np.ones(n_src) / n_src
    nu = np.ones(n_tgt) / n_tgt

    cost = pairwise_l2(source, target).numpy()

    try:
        plan = ot.emd(mu, nu, cost)
    except Exception as e:
        log(f"      Fallo en ot.emd: {e}")
        return None, None

    plan_t = torch.from_numpy(plan).float()

    X_prime = barycentric_projection(plan_t, target)

    return plan_t, X_prime


def _process_fold(clf, X_val, y_val, M, M_inv_T, chain_order) -> dict | None:
    y_pred = clf.predict(X_val.numpy())
    rej_mask = y_pred == 0
    acc_mask = y_pred == 1
    X_rej = X_val[rej_mask]
    X_acc = X_val[acc_mask]

    if X_rej.shape[0] < 5 or X_acc.shape[0] < 5:
        return None

    rng = np.random.RandomState(42)
    if X_rej.shape[0] > MAX_SAMPLES:
        keep = rng.choice(X_rej.shape[0], MAX_SAMPLES, replace=False)
        X_rej = X_rej[keep]
    if X_acc.shape[0] > MAX_SAMPLES:
        keep = rng.choice(X_acc.shape[0], MAX_SAMPLES, replace=False)
        X_acc = X_acc[keep]

    U_rej = X_rej @ M.T
    U_acc = X_acc @ M.T

    C_ambient = pairwise_l2(X_rej, X_acc)
    C_latent = pairwise_l2(U_rej, U_acc)

    plan_std, X_std = compute_transport(X_rej, X_acc)
    plan_causal, U_causal_prime = compute_transport(U_rej, U_acc)

    X_causal = U_causal_prime @ M_inv_T if U_causal_prime is not None else None

    fold_results = {}
    for method, plan, X_prime in [
        ("Standard OT", plan_std, X_std),
        ("Causal OT", plan_causal, X_causal),
    ]:
        if plan is None or X_prime is None:
            fold_results[f"{method}_validity"] = float("nan")
            fold_results[f"{method}_ambient_cost"] = float("nan")
            fold_results[f"{method}_latent_effort"] = float("nan")
            continue

        N_rej = X_rej.shape[0]
        ambient_cost = (plan * C_ambient).sum().item()
        latent_effort = (plan * C_latent).sum().item()

        prime_cls = clf.predict(X_prime.numpy())
        validity = (prime_cls == 1).mean() * 100

        fold_results[f"{method}_validity"] = validity
        fold_results[f"{method}_ambient_cost"] = ambient_cost
        fold_results[f"{method}_latent_effort"] = latent_effort

    return fold_results


DATASET_CONFIGS = {
    "adult": {
        "chain_cols": ["age", "education-num", "hours-per-week"],
        "label": "Adult",
    },
    "lsac": {
        "chain_cols": ["fam_inc", "ugpa", "lsat"],
        "label": "LSAC",
    },
    "compas": {
        "chain_cols": ["age", "juv_crimes", "priors_count"],
        "label": "COMPAS",
    },
    "german": {
        "chain_cols": ["age", "credit_amount", "duration"],
        "label": "German",
    },
}


def prepare_dataset(ds_name: str, cfg: dict):
    df, target, info = load_dataset(ds_name)
    cols = cfg["chain_cols"]
    if ds_name == "compas":
        df["juv_crimes"] = df["juv_fel_count"] + df["juv_misd_count"] + df["juv_other_count"]
    X_df = df[cols].copy()
    for c in X_df.columns:
        if X_df[c].dtype.name == "category":
            X_df[c] = X_df[c].cat.codes.astype(np.float32)
    X = (X_df - X_df.mean()) / X_df.std()
    X_ten = torch.from_numpy(X.values.astype(np.float32))
    y_np = target.values.astype(int)
    return X_ten, y_np, X_df.columns.tolist()


def run() -> pd.DataFrame:
    log("#" * 60)
    log("C2: The Cost of Causal Ignorance — Latent Intervention Effort (CV)")
    log("#" * 60)

    all_rows = []
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

    for ds_name, cfg in DATASET_CONFIGS.items():
        log(f"\n=== {cfg['label']} ===")
        X, y, col_names = prepare_dataset(ds_name, cfg)
        log(f"  Loaded {X.shape[0]} samples, {X.shape[1]} features")

        chain_order = list(range(X.shape[1]))

        valid_folds = 0
        for fold, (train_idx, val_idx) in enumerate(kf.split(X.numpy())):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            clf = RandomForestClassifier(n_estimators=100, random_state=42 + fold)
            clf.fit(X_train.numpy(), y_train)

            M, M_inv_T = fit_linear_scm(X_val, chain_order)
            kappa = compute_anisotropy(M)

            try:
                fold_res = _process_fold(clf, X_val, y_val, M, M_inv_T, chain_order)
            except Exception as e:
                log(f"    Fold {fold}: ERROR — {e}")
                continue
            if fold_res is None:
                log(f"    Fold {fold}: SKIP (too few samples in rej or acc)")
                continue

            y_pred = clf.predict(X_val.numpy())
            n_rej = int((y_pred == 0).sum())
            n_acc = int((y_pred == 1).sum())
            fold_res["dataset"] = cfg["label"]
            fold_res["kappa"] = kappa
            fold_res["fold"] = fold
            all_rows.append(fold_res)
            valid_folds += 1
            log(f"    Fold {fold}: rej={n_rej} acc={n_acc}, "
                f"validity_std={fold_res['Standard OT_validity']:.1f}%, "
                f"validity_causal={fold_res['Causal OT_validity']:.1f}%")

        log(f"  Completed {valid_folds}/{N_FOLDS} valid folds")

    df = pd.DataFrame(all_rows)
    path = RESULTS_DIR / "c2_recourse_causal.csv"
    df.to_csv(path, index=False)
    log(f"\nResults saved to {path} ({len(df)} rows)")
    return df


if __name__ == "__main__":
    run()
