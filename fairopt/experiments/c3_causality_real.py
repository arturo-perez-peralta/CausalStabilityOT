"""
C3: Causal OT on real / realistic data.

Two parts:
  * ``run_ecoli70`` — Standard OT vs Causal OT on data simulated from the
    ecoli70 Bayesian network (46 nodes). The binary label is used ONLY to
    split into source (rejected) / target (accepted) distributions for OT.
  * ``run`` — conjugacy deviation between the ambient (X) and latent (U, PCA)
    spaces across lsac / student / credit_default with 10-fold CV.
"""
from __future__ import annotations

import torch
import pandas as pd
import numpy as np
import gzip
import shutil
import tempfile
import urllib.request
from sklearn.decomposition import PCA
from tqdm import tqdm
from pathlib import Path

from typing import Optional, List

from fairopt.experiments.runner import prepare_data, save_results, Timer
from fairopt.experiments.config import get_config
from fairopt.core.cost import pairwise_l2
from fairopt.core.sinkhorn import SinkhornSolver
from fairopt.core.metrics import barycentric_projection
import ot

DATASETS = ["lsac", "student", "credit_default"]
MAX_SAMPLES = None
N_COMPONENTS = 3

RESULTS_DIR = Path(__file__).resolve().parents[2] / "results"


def load_ecoli70_bn() -> tuple:
    """Load ecoli70 Gaussian BN from bnlearn repository via rdata.

    Returns:
        node_names: list of str, names of the 46 nodes
        B: np.ndarray (46, 46), weighted adjacency (B[i,j] = coef of parent j -> child i)
        intercepts: np.ndarray (46,), regression intercepts
        std_devs: np.ndarray (46,), conditional standard deviations
        M: np.ndarray (46, 46), linear map from latent noise to observations X = M @ U
    """
    url = "https://www.bnlearn.com/bnrepository/ecoli70/ecoli70.rda"
    tmp_dir = Path(tempfile.gettempdir())
    rda_path = tmp_dir / "ecoli70.rda"
    decompressed_path = tmp_dir / "ecoli70_decompressed.rda"

    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as resp:
        with open(rda_path, "wb") as f:
            f.write(resp.read())

    with gzip.open(rda_path, "rb") as f_in:
        with open(decompressed_path, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)

    import rdata
    parsed = rdata.read_rda(str(decompressed_path))
    bn = parsed["bn"]
    node_names = [str(name) for name in bn.keys()]
    n_nodes = len(node_names)

    B = np.zeros((n_nodes, n_nodes))
    intercepts = np.zeros(n_nodes)
    std_devs = np.zeros(n_nodes)

    for i, name in enumerate(node_names):
        nd = bn[name]
        if hasattr(nd["coefficients"], "values"):
            coefs = nd["coefficients"].values
        else:
            coefs = np.asarray(nd["coefficients"]).flatten()
        parents = [str(p) for p in nd["parents"]]
        sd = float(np.asarray(nd["sd"]).flat[0])
        std_devs[i] = sd
        intercepts[i] = coefs[0]
        for j, pname in enumerate(parents):
            pidx = node_names.index(pname)
            B[i, pidx] = coefs[j + 1]

    I = np.eye(n_nodes)
    M = np.linalg.solve(I - B, np.diag(std_devs))

    return node_names, B, intercepts, std_devs, M


def _compute_transport(source: torch.Tensor, target: torch.Tensor) -> torch.Tensor | None:
    n_src = source.shape[0]
    n_tgt = target.shape[0]

    mu = np.ones(n_src) / n_src
    nu = np.ones(n_tgt) / n_tgt

    cost = pairwise_l2(source, target).numpy()

    try:
        plan = ot.emd(mu, nu, cost)
    except Exception as e:
        print(f"      Fallo en ot.emd: {e}")
        return None

    return torch.from_numpy(plan).float()


def run_ecoli70(n_obs: int = 5000, sinkhorn_eps: float = 1.0) -> pd.DataFrame:
    """Standard OT vs Causal OT on data simulated from the ecoli70 BN.

    Generates synthetic data from the ecoli70 Bayesian network (46 nodes):
      X = M @ U,  U ~ N(0, I)
    where M is the linear map from latent noise to observations.
    The binary label y = 1{terminal_node > median} is used ONLY to split
    into source (rejected, y=0) and target (accepted, y=1) distributions.
    The label plays NO role in the causal graph — it is purely for
    generating the source/target split for OT transport comparison.
    """
    from sklearn.model_selection import KFold
    from sklearn.ensemble import RandomForestClassifier

    log_path = RESULTS_DIR / "c3_run.log"
    def log(msg):
        print(msg, flush=True)
        with open(log_path, "a") as f:
            f.write(msg + "\n")

    log("#" * 60)
    log("C3 — ecoli70 BN simulation")
    log("#" * 60)

    node_names, B, intercepts, std_devs, M = load_ecoli70_bn()
    d_bn = M.shape[0]
    M_t = torch.from_numpy(M).float()
    M_inv_t = torch.linalg.inv(M_t)

    G = M.T @ M
    evals = np.linalg.eigvalsh(G)
    kappa = float(evals.max() / evals.min())
    log(f"  ecoli70 BN: d={d_bn}, edges={int(np.sum(B != 0))}, kappa={kappa:.4f}")

    # Generate synthetic data from the BN
    rng = np.random.RandomState(42)
    U = torch.randn(n_obs, d_bn)
    X = U @ M_t.T

    # Binary label: indicator of terminal node above median.
    # The terminal node is the last node in topological order (no children).
    # This label is used ONLY for rej/acc split, not in the causal graph.
    terminal_idx = len(node_names) - 1
    y = (X[:, terminal_idx] > X[:, terminal_idx].median()).numpy().astype(int)
    n_pos = int(y.sum())
    n_neg = n_obs - n_pos
    log(f"  Generated {n_obs} samples: {n_neg} rejected (y=0), {n_pos} accepted (y=1)")
    log(f"  Terminal node: '{node_names[terminal_idx]}'")

    all_rows = []
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    for fold, (train_idx, val_idx) in enumerate(kf.split(X.numpy())):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        clf = RandomForestClassifier(n_estimators=100, random_state=42 + fold)
        clf.fit(X_train.numpy(), y_train)
        y_pred = clf.predict(X_val.numpy())

        rej_mask = y_pred == 0
        acc_mask = y_pred == 1
        X_rej = X_val[rej_mask]
        X_acc = X_val[acc_mask]

        if X_rej.shape[0] < 5 or X_acc.shape[0] < 5:
            log(f"    Fold {fold}: SKIP (rej={X_rej.shape[0]}, acc={X_acc.shape[0]})")
            continue

        if X_rej.shape[0] > 2000:
            keep = rng.choice(X_rej.shape[0], 2000, replace=False)
            X_rej = X_rej[keep]
        if X_acc.shape[0] > 2000:
            keep = rng.choice(X_acc.shape[0], 2000, replace=False)
            X_acc = X_acc[keep]

        U_rej = X_rej @ M_inv_t.T
        U_acc = X_acc @ M_inv_t.T

        # 1. Calculamos los planes de transporte pi exactos
        plan_std = _compute_transport(X_rej, X_acc)
        plan_causal = _compute_transport(U_rej, U_acc)

        # 2. Matrices de coste exactas
        C_ambient = pairwise_l2(X_rej, X_acc)
        C_latent = pairwise_l2(U_rej, U_acc)

        # Puntos de destino reconstruidos para el OT Causal (de U a X)
        X_causal_targets = U_acc @ M_t.T

        fold_res = {"dataset": "ecoli70", "kappa": kappa, "fold": fold}

        for method, plan, target_points, C_matrix in [
            ("Standard OT", plan_std, X_acc, C_ambient),
            ("Causal OT", plan_causal, X_causal_targets, C_latent),
        ]:
            if plan is None:
                fold_res[f"{method}_validity"] = float("nan")
                fold_res[f"{method}_ambient_cost"] = float("nan")
                fold_res[f"{method}_latent_effort"] = float("nan")
                continue

            N_rej = X_rej.shape[0]

            # 3. Costes de Kantorovich (esperanza matemática del coste por individuo)
            ambient_cost = (plan * C_ambient).sum().item()
            latent_effort = (plan * C_latent).sum().item()

            # 4. Validez esperada
            target_preds = clf.predict(target_points.numpy())
            target_preds_tensor = torch.from_numpy(target_preds).float()

            # Probabilidad esperada de terminar en un estado válido (y=1)
            expected_validity = (plan @ target_preds_tensor).sum().item() * N_rej * 100

            fold_res[f"{method}_validity"] = expected_validity
            fold_res[f"{method}_ambient_cost"] = ambient_cost
            fold_res[f"{method}_latent_effort"] = latent_effort

        all_rows.append(fold_res)
        log(f"    Fold {fold}: rej={int(rej_mask.sum())} acc={int(acc_mask.sum())}, "
            f"validity_std={fold_res['Standard OT_validity']:.1f}%, "
            f"validity_causal={fold_res['Causal OT_validity']:.1f}%")

    df = pd.DataFrame(all_rows)
    path = RESULTS_DIR / "c3_causality_ecoli70.csv"
    df.to_csv(path, index=False)
    log(f"\nResults saved to {path} ({len(df)} rows)")
    return df


def compute_conjugacy_diff(
    X_s: torch.Tensor, X_t: torch.Tensor,
    M: torch.Tensor, U_s: torch.Tensor, U_t: torch.Tensor,
    sinkhorn_eps: float = 1.0,
) -> float:
    n_src = X_s.shape[0]
    n_tgt = X_t.shape[0]
    mu = torch.ones(n_src) / n_src
    nu = torch.ones(n_tgt) / n_tgt

    solver = SinkhornSolver(epsilon=sinkhorn_eps, max_iter=500, tol=1e-6)

    cost_X = pairwise_l2(X_s, X_t)
    res_X = solver.solve(mu, nu, cost_X)
    if res_X.plan is None:
        return float("nan")
    proj_X = barycentric_projection(res_X.plan, X_t)

    cost_U = pairwise_l2(U_s, U_t)
    res_U = solver.solve(mu, nu, cost_U)
    if res_U.plan is None:
        return float("nan")
    proj_U = barycentric_projection(res_U.plan, U_t)

    proj_U_obs = proj_U @ M.T
    diff = proj_X - proj_U_obs
    return (diff ** 2).sum(dim=1).mean().item()


def run(device: str = "cpu", use_cv: bool = False) -> pd.DataFrame:
    all_rows = []

    for dataset_name in DATASETS:
        cfg = get_config(dataset_name)
        data = prepare_data(
            dataset_name,
            numerical_cols=cfg["numerical_cols"],
            categorical_cols=cfg["categorical_cols"],
            sensitive_cols=cfg["sensitive_single"],
            max_samples=MAX_SAMPLES,
        )
        X_np = data["X"].cpu().numpy()

        pca = PCA(n_components=min(N_COMPONENTS, X_np.shape[1], X_np.shape[0]))
        U_np = pca.fit_transform(X_np)
        M_mat = pca.components_.T
        d_latent = U_np.shape[1]
        d_obs = X_np.shape[1]

        U = torch.from_numpy(U_np).float()
        X = torch.from_numpy(X_np).float()
        M_t = torch.from_numpy(M_mat).float()
        n = X.shape[0]

        if use_cv:
            n_folds = 10
            fold_size = n // n_folds
            perm = torch.randperm(n)
            splits = []
            for fold in range(n_folds):
                val_idx = perm[fold * fold_size : (fold + 1) * fold_size]
                train_idx = torch.cat([perm[: fold * fold_size], perm[(fold + 1) * fold_size :]])
                splits.append((train_idx, val_idx))
            it = tqdm(splits, desc=f"{dataset_name} CV 10-fold")
        else:
            splits = [(data["train_idx"], data["test_idx"])]
            it = splits

        diffs = []
        for fold_or_idx, (train_idx, val_idx) in enumerate(it):
            X_s, X_t = X[train_idx], X[val_idx]
            U_s, U_t = U[train_idx], U[val_idx]

            diff = compute_conjugacy_diff(X_s, X_t, M_t, U_s, U_t)
            diffs.append(diff)

        diffs = [d for d in diffs if not np.isnan(d)]
        if diffs:
            all_rows.append({
                "dataset": dataset_name,
                "n_components": d_latent,
                "n_obs": d_obs,
                "diff_mean": float(np.mean(diffs)),
                "diff_std": float(np.std(diffs, ddof=1)) if len(diffs) > 1 else 0.0,
                "n_folds": len(diffs),
                "explained_var_ratio": pca.explained_variance_ratio_[:d_latent].sum(),
            })

    df = pd.DataFrame(all_rows)
    save_results(df, "c3_causality_real")
    return df


if __name__ == "__main__":
    run_ecoli70()
