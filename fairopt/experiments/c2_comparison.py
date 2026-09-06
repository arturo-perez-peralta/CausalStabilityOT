import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import gzip
import shutil
import urllib.request
import time
import argparse

from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import KFold
from scipy.stats import gaussian_kde
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
import gurobipy as gp
from gurobipy import GRB
from pathlib import Path
import ot
import rdata
from joblib import Parallel, delayed

try:
    from fairopt.data.datasets import load_dataset
except ImportError:
    print("Warning: Could not import fairopt.")

NUM_CORES = "1"
os.environ["OMP_NUM_THREADS"] = NUM_CORES
os.environ["OPENBLAS_NUM_THREADS"] = NUM_CORES
os.environ["MKL_NUM_THREADS"] = NUM_CORES
os.environ["VECLIB_MAXIMUM_THREADS"] = NUM_CORES
os.environ["NUMEXPR_NUM_THREADS"] = NUM_CORES
torch.set_num_threads(int(NUM_CORES))

BNS = ["ecoli70", "magic-niab", "magic-irri", "arth150"]

REAL_DATASETS_CFG = {
    "german": {"chain_cols": ["age", "credit_amount", "duration"], "label": "German"},
    "compas": {"chain_cols": ["age", "juv_crimes", "priors_count"], "label": "COMPAS"},
    "adult": {"chain_cols": ["age", "education-num", "hours-per-week"], "label": "Adult"},
    "lsac": {"chain_cols": ["fam_inc", "ugpa", "lsat"], "label": "LSAC"},
}

ALL_DATASETS = BNS + list(REAL_DATASETS_CFG.keys())
ALL_BASELINES = ["Standard OT", "Causal OT", "Wachter", "Karimi", "Karimi_Hard", "FACEGroup", "Carrizosa_Exact"]

class MLP(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 1)
        )
    def forward(self, x):
        return self.net(x).squeeze(-1)

def load_bn(name: str) -> tuple:
    url = f"https://www.bnlearn.com/bnrepository/{name}/{name}.rda"
    rda_path = Path(f"/tmp/{name}.rda")
    decompressed_path = Path(f"/tmp/{name}_decompressed.rda")

    if not rda_path.exists():
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req) as resp:
            with open(rda_path, "wb") as f:
                f.write(resp.read())

    if not decompressed_path.exists():
        with gzip.open(rda_path, "rb") as f_in:
            with open(decompressed_path, "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)

    parsed = rdata.read_rda(str(decompressed_path))
    bn = parsed["bn"]
    node_names = [str(k) for k in bn.keys()]
    n_nodes = len(node_names)

    B = np.zeros((n_nodes, n_nodes))
    intercepts, std_devs = np.zeros(n_nodes), np.zeros(n_nodes)

    for i, n_name in enumerate(node_names):
        nd = bn[n_name]
        coefs = nd["coefficients"].values if hasattr(nd["coefficients"], "values") else np.asarray(nd["coefficients"]).flatten()
        parents = [str(p) for p in nd["parents"]]
        std_devs[i] = float(np.asarray(nd["sd"]).flat[0])
        intercepts[i] = coefs[0]
        for j, pname in enumerate(parents):
            B[i, node_names.index(pname)] = coefs[j + 1]

    I = np.eye(n_nodes)
    M = np.linalg.solve(I - B, np.diag(std_devs))
    
    out_degree = (B != 0).sum(axis=0)
    terminal_candidates = np.where(out_degree == 0)[0]
    terminal_idx = terminal_candidates[np.argmax([(M[idx] != 0).sum() for idx in terminal_candidates])]

    return node_names, B, intercepts, std_devs, M, terminal_idx

def fit_linear_scm(X: torch.Tensor, chain_order) -> tuple:
    p = X.shape[1]
    X_np = X.numpy()
    M_x_to_u = np.eye(p)
    for k in range(1, p):
        parent_idx, child_idx = chain_order[k - 1], chain_order[k]
        beta = np.linalg.lstsq(X_np[:, parent_idx][:, None], X_np[:, child_idx], rcond=None)[0][0]
        M_x_to_u[child_idx, parent_idx] = -beta
    
    M_x_to_u_t = torch.from_numpy(M_x_to_u).float()
    M_u_to_x_t = torch.linalg.inv(M_x_to_u_t) 
    return M_u_to_x_t, M_x_to_u_t

def compute_anisotropy(M: np.ndarray) -> float:
    G = M.T @ M
    evals = np.linalg.eigvalsh(G)
    return float(evals.max() / evals.min())

def get_dataset_data(ds_name: str, n_obs: int = 10000):
    if ds_name in BNS:
        node_names, B, intercepts, std_devs, M, terminal_idx = load_bn(ds_name)
        d = M.shape[0]
        M_t = torch.from_numpy(M).float()
        M_inv_t = torch.linalg.inv(M_t)

        rng = np.random.RandomState(42)
        X = torch.randn(n_obs, d) @ M_t.T
        y = (X[:, terminal_idx] > X[:, terminal_idx].median()).numpy().astype(float)
        kappa = compute_anisotropy(M)
        label = ds_name
    elif ds_name in REAL_DATASETS_CFG:
        cfg = REAL_DATASETS_CFG[ds_name]
        label = cfg["label"]
        df, target, _ = load_dataset(ds_name)
        if ds_name == "compas": df["juv_crimes"] = df["juv_fel_count"] + df["juv_misd_count"] + df["juv_other_count"]
        
        X_df = df[cfg["chain_cols"]].copy()
        for c in X_df.columns:
            if X_df[c].dtype.name == "category": X_df[c] = X_df[c].cat.codes.astype(np.float32)
                
        X_np = ((X_df - X_df.mean()) / X_df.std()).values.astype(np.float32)
        X, y = torch.from_numpy(X_np), target.values.astype(float)
        
        M_t, M_inv_t = fit_linear_scm(X, list(range(X.shape[1])))
        kappa = compute_anisotropy(M_t.numpy())
    return X, torch.from_numpy(y).float(), M_t, M_inv_t, kappa, label

def get_ot_plan_and_cost(source: torch.Tensor, target: torch.Tensor):
    n_src, n_tgt = source.shape[0], target.shape[0]
    mu, nu = np.ones(n_src)/n_src, np.ones(n_tgt)/n_tgt
    cost = ((source.unsqueeze(1) - target.unsqueeze(0)) ** 2).sum(dim=2).numpy()
    plan = ot.emd(mu, nu, cost)
    return plan, cost

def calculate_pi_weighted_cost(plan: np.ndarray, cost: np.ndarray, n_src: int) -> float:
    expected_cost_per_instance = ((plan * n_src) * cost).sum(axis=1)
    return float(expected_cost_per_instance.mean())

def get_knn_dist_vector(source_proj: torch.Tensor, target: torch.Tensor, k: int = 5) -> np.ndarray:
    nbrs = NearestNeighbors(n_neighbors=min(k, target.shape[0]), algorithm='auto').fit(target.numpy())
    distances, _ = nbrs.kneighbors(source_proj.numpy())
    return distances.mean(axis=1)

def compute_knn_distance(source_proj: torch.Tensor, target: torch.Tensor, k: int = 5) -> float:
    return float(get_knn_dist_vector(source_proj, target, k).mean())

def get_barycentric_proj(plan: np.ndarray, target_tensor: torch.Tensor) -> torch.Tensor:
    mass_i = plan.sum(axis=1)
    valid_mask = mass_i > 1e-12
    pi_cond = np.zeros_like(plan)
    pi_cond[valid_mask] = plan[valid_mask] / mass_i[valid_mask][:, None]
    return torch.from_numpy(np.matmul(pi_cond, target_tensor.numpy())).float()

def compute_extra_metrics(X_cf_or_plan, X_acc: torch.Tensor, model: nn.Module, is_ot: bool = False):
    X_acc_np = X_acc.cpu().detach().numpy()
    
    with torch.no_grad():
        probs_acc = torch.sigmoid(model(X_acc)).cpu().numpy()
        
    if is_ot:
        plan = X_cf_or_plan
        mass_i = plan.sum(axis=1)
        valid = mass_i > 1e-12
        pi_cond = np.zeros_like(plan)
        pi_cond[valid] = plan[valid] / mass_i[valid][:, None]
        
        conf_i = np.sum(pi_cond * probs_acc[None, :], axis=1)
        confidence = conf_i[valid].mean() * 100 if valid.sum() > 0 else 0.0
        
        if valid.sum() > 0:
            proj_X_np = np.matmul(pi_cond[valid], X_acc_np)
            a, b = np.ones(len(proj_X_np)) / len(proj_X_np), np.ones(len(X_acc_np)) / len(X_acc_np)
            M = ot.dist(proj_X_np, X_acc_np, metric='sqeuclidean')
            wass_dist = float(ot.emd2(a, b, M))
        else:
            wass_dist = float('nan')
    else:
        X_cf = X_cf_or_plan
        with torch.no_grad():
            confidence = torch.sigmoid(model(X_cf)).mean().item() * 100
        
        X_cf_np = X_cf.cpu().detach().numpy()
        a, b = np.ones(len(X_cf_np)) / len(X_cf_np), np.ones(len(X_acc_np)) / len(X_acc_np)
        M = ot.dist(X_cf_np, X_acc_np, metric='sqeuclidean')
        wass_dist = float(ot.emd2(a, b, M))
        
    return confidence, wass_dist

def run_wachter(X_rej: torch.Tensor, model: nn.Module) -> torch.Tensor:
    X_cf = X_rej.clone().detach()
    final_X_cf = X_rej.clone().detach()
    is_successful = torch.zeros(X_rej.shape[0], dtype=torch.bool, device=X_rej.device)
    lambdas = torch.ones(X_rej.shape[0], device=X_rej.device) * 0.01
    
    for step in range(5):
        current_X_cf = X_cf.clone().detach().requires_grad_(True)
        optimizer = optim.Adam([current_X_cf], lr=0.1)
        target = torch.ones(X_rej.shape[0], device=X_rej.device)
        
        for _ in range(150):
            optimizer.zero_grad()
            mse = ((current_X_cf - X_rej) ** 2).sum(dim=1)
            bce = torch.nn.functional.binary_cross_entropy_with_logits(model(current_X_cf), target, reduction='none')
            loss = (mse + lambdas * bce).mean()
            loss.backward()
            optimizer.step()
            
        with torch.no_grad():
            preds = model(current_X_cf) > 0
            new_success = preds & ~is_successful
            final_X_cf[new_success] = current_X_cf[new_success].detach()
            is_successful = is_successful | preds
            if is_successful.all(): break
            lambdas[~is_successful] *= 10.0
            X_cf = current_X_cf.detach() 
            
    final_X_cf[~is_successful] = current_X_cf[~is_successful].detach()
    return final_X_cf

def run_karimi(X_rej: torch.Tensor, M_t: torch.Tensor, M_inv_t: torch.Tensor, model: nn.Module) -> torch.Tensor:
    U_rej = X_rej @ M_inv_t.T
    U_cf = U_rej.clone().detach()
    final_X_cf = X_rej.clone().detach()
    is_successful = torch.zeros(U_rej.shape[0], dtype=torch.bool, device=X_rej.device)
    lambdas = torch.ones(U_rej.shape[0], device=X_rej.device) * 0.01
    
    for step in range(5):
        delta = torch.zeros_like(U_cf, requires_grad=True)
        optimizer = optim.Adam([delta], lr=0.1)
        target = torch.ones(U_rej.shape[0], device=X_rej.device)
        
        for _ in range(150):
            optimizer.zero_grad()
            current_U = U_cf + delta
            mse = ((current_U - U_rej) ** 2).sum(dim=1)
            current_X = current_U @ M_t.T
            bce = torch.nn.functional.binary_cross_entropy_with_logits(model(current_X), target, reduction='none')
            loss = (mse + lambdas * bce).mean()
            loss.backward()
            optimizer.step()
            
        with torch.no_grad():
            current_X = (U_cf + delta) @ M_t.T
            preds = model(current_X) > 0
            new_success = preds & ~is_successful
            final_X_cf[new_success] = current_X[new_success].detach()
            is_successful = is_successful | preds
            if is_successful.all(): break
            lambdas[~is_successful] *= 10.0
            U_cf = (U_cf + delta).detach()
            
    final_X_cf[~is_successful] = current_X[~is_successful].detach()
    return final_X_cf

def run_facegroup(X_rej, X_acc):
    X_all = np.vstack([X_rej.numpy(), X_acc.numpy()])
    kde, densities = gaussian_kde(X_all.T), gaussian_kde(X_all.T)(X_all.T)
    distances, indices = NearestNeighbors(n_neighbors=min(10, X_all.shape[0])).fit(X_all).kneighbors(X_all)
    
    row, col, data = [], [], []
    for i in range(X_all.shape[0]):
        for j, d in zip(indices[i], distances[i]):
            if i != j:
                row.append(i); col.append(j); data.append(d / (min(densities[i], densities[j]) + 1e-6))
                
    dist_matrix = dijkstra(csgraph=csr_matrix((data, (row, col)), shape=(X_all.shape[0], X_all.shape[0])), directed=False, indices=np.arange(len(X_rej)))[:, len(X_rej):]
    eps, covered, centers, assigned = np.percentile(np.min(dist_matrix, axis=1), 75), set(), [], np.zeros(len(X_rej), dtype=int)
    
    for _ in range(min(20, len(X_acc))):
        best_cov = []
        for v in range(len(X_acc)):
            cov = [u for u in range(len(X_rej)) if u not in covered and dist_matrix[u, v] <= eps]
            if len(cov) > len(best_cov): best_v, best_cov = v, cov
        if not best_cov: break
        centers.append(best_v)
        for u in best_cov: covered.add(u); assigned[u] = best_v
            
    for u in range(len(X_rej)):
        if u not in covered: assigned[u] = centers[np.argmin(dist_matrix[u, centers])] if centers else np.argmin(dist_matrix[u])
    return torch.from_numpy(X_acc.numpy()[assigned]).float()

def solve_single_carrizosa(x_0_np, d, W1, b1, W2, b2, W3, b3):
    env = gp.Env(empty=True)
    env.setParam("OutputFlag", 0)
    env.setParam("Threads", 1) 
    env.start()
    m = gp.Model("c_exact", env=env)
    m.setParam('TimeLimit', 10)
    
    x_prime = m.addMVar(shape=d, lb=-GRB.INFINITY)
    M = 100.0
    
    z1, h1, a1 = m.addMVar(128, lb=-GRB.INFINITY), m.addMVar(128, lb=0.0), m.addMVar(128, vtype=GRB.BINARY)
    m.addConstr(z1 == W1 @ x_prime + b1); m.addConstr(h1 >= z1); m.addConstr(h1 <= M * a1); m.addConstr(h1 <= z1 + M * (1 - a1))
    z2, h2, a2 = m.addMVar(64, lb=-GRB.INFINITY), m.addMVar(64, lb=0.0), m.addMVar(64, vtype=GRB.BINARY)
    m.addConstr(z2 == W2 @ h1 + b2); m.addConstr(h2 >= z2); m.addConstr(h2 <= M * a2); m.addConstr(h2 <= z2 + M * (1 - a2))
    
    out = m.addMVar(1, lb=-GRB.INFINITY)
    m.addConstr(out == W3 @ h2 + b3)
    m.addConstr(out >= 0.01)
    
    m.setObjective((x_prime - x_0_np) @ (x_prime - x_0_np), GRB.MINIMIZE)
    m.optimize()
    return torch.from_numpy(x_prime.X).float() if m.status in [GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.SUBOPTIMAL] and m.SolCount > 0 else torch.from_numpy(x_0_np).float()

def solve_single_karimi_hard(x_0_np, u_0_np, B_np, d, W1, b1, W2, b2, W3, b3):
    env = gp.Env(empty=True)
    env.setParam("OutputFlag", 0)
    env.setParam("Threads", 1) 
    env.start()
    m = gp.Model("k_hard", env=env)
    m.setParam('TimeLimit', 10)
    
    x_prime = m.addMVar(shape=d, lb=-GRB.INFINITY)
    m_vars = m.addMVar(shape=d, vtype=GRB.BINARY) 
    M = 100.0
    
    z1, h1, a1 = m.addMVar(128, lb=-GRB.INFINITY), m.addMVar(128, lb=0.0), m.addMVar(128, vtype=GRB.BINARY)
    m.addConstr(z1 == W1 @ x_prime + b1); m.addConstr(h1 >= z1); m.addConstr(h1 <= M * a1); m.addConstr(h1 <= z1 + M * (1 - a1))
    z2, h2, a2 = m.addMVar(64, lb=-GRB.INFINITY), m.addMVar(64, lb=0.0), m.addMVar(64, vtype=GRB.BINARY)
    m.addConstr(z2 == W2 @ h1 + b2); m.addConstr(h2 >= z2); m.addConstr(h2 <= M * a2); m.addConstr(h2 <= z2 + M * (1 - a2))
    
    out = m.addMVar(1, lb=-GRB.INFINITY)
    m.addConstr(out == W3 @ h2 + b3)
    m.addConstr(out >= 0.01)
    
    scm_eq = (np.eye(d) - B_np) @ x_prime - u_0_np
    m.addConstr(scm_eq <= M * m_vars)
    m.addConstr(scm_eq >= -M * m_vars)
    
    diff = x_prime - x_0_np
    m.setObjective(diff @ diff + 10.0 * m_vars.sum(), GRB.MINIMIZE)
    m.optimize()
    return torch.from_numpy(x_prime.X).float() if m.status in [GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.SUBOPTIMAL] and m.SolCount > 0 else torch.from_numpy(x_0_np).float()

def run_gurobi_baselines(X_rej: torch.Tensor, M_inv_t: torch.Tensor, model: nn.Module, method="carrizosa"):
    X_rej_np = X_rej.numpy()
    d = X_rej_np.shape[1]
    W1, b1 = model.net[0].weight.data.numpy(), model.net[0].bias.data.numpy()
    W2, b2 = model.net[2].weight.data.numpy(), model.net[2].bias.data.numpy()
    W3, b3 = model.net[4].weight.data.numpy(), model.net[4].bias.data.numpy()
    N_JOBS = 8
    
    if method == "carrizosa":
        res = Parallel(n_jobs=N_JOBS)(delayed(solve_single_carrizosa)(X_rej_np[i], d, W1, b1, W2, b2, W3, b3) for i in range(X_rej.shape[0]))
    else: 
        B_np = (torch.eye(d) - M_inv_t).numpy()
        U_rej_np = (X_rej @ M_inv_t.T).numpy()
        res = Parallel(n_jobs=N_JOBS)(delayed(solve_single_karimi_hard)(X_rej_np[i], U_rej_np[i], B_np, d, W1, b1, W2, b2, W3, b3) for i in range(X_rej.shape[0]))
    return torch.stack(res)

def run_unified_experiments(datasets_to_run=None, baselines_to_run=None, n_obs=10000):
    datasets_to_run = datasets_to_run or ALL_DATASETS
    baselines_to_run = baselines_to_run or ALL_BASELINES

    all_rows = []
    for ds_name in datasets_to_run:
        print(f"\n[{ds_name}] Preparing dataset...")
        try: X, y, M_t, M_inv_t, kappa, label = get_dataset_data(ds_name, n_obs)
        except Exception as e: print(f"[{ds_name}] Error: {e}"); continue

        d = X.shape[1]
        for fold, (train_idx, val_idx) in enumerate(KFold(n_splits=5, shuffle=True, random_state=42).split(X.numpy())):
            X_train, X_val, y_train, y_val = X[train_idx], X[val_idx], y[train_idx], y[val_idx]

            model = MLP(d)
            optimizer, criterion = optim.Adam(model.parameters(), lr=0.01), nn.BCEWithLogitsLoss()
            for _ in range(100):
                optimizer.zero_grad()
                loss = criterion(model(X_train), y_train)
                loss.backward(); optimizer.step()
                
            model.eval()
            with torch.no_grad():
                preds_val = (model(X_val) > 0).float()
                acc = (preds_val == y_val).float().mean().item()

            print(f"[{label} - Fold {fold}] Acc: {acc:.4f} | Kappa: {kappa:.4f}")
            X_rej, X_acc = X_val[preds_val == 0], X_val[preds_val == 1]
            if X_rej.shape[0] < 5 or X_acc.shape[0] < 5: continue

            n_src = X_rej.shape[0]
            U_rej = X_rej @ M_inv_t.T
            U_acc = X_acc @ M_inv_t.T

            fold_res = {"dataset": label, "fold": fold, "kappa": kappa, "accuracy": acc}

            if "Standard OT" in baselines_to_run or "Causal OT" in baselines_to_run:
                knn_dists_X_acc = get_knn_dist_vector(X_acc, X_acc, k=5)
                cost_knn = np.tile(knn_dists_X_acc, (n_src, 1))

            if "Standard OT" in baselines_to_run:
                t0 = time.time()
                plan_std, cost_X = get_ot_plan_and_cost(X_rej, X_acc)
                cost_U = ((U_rej.unsqueeze(1) - U_acc.unsqueeze(0)) ** 2).sum(dim=2).numpy()
                conf_std, wass_std = compute_extra_metrics(plan_std, X_acc, model, is_ot=True)
                
                # Usamos la proyección baricéntrica para computar la validez de OT estándar
                proj_X_std = get_barycentric_proj(plan_std, X_acc)
                validity_std = (model(proj_X_std) > 0).float().mean().item() * 100
                
                fold_res.update({
                    "Standard OT_validity_real": validity_std, 
                    "Standard OT_ambient_cost": calculate_pi_weighted_cost(plan_std, cost_X, n_src),
                    "Standard OT_latent_effort": calculate_pi_weighted_cost(plan_std, cost_U, n_src),
                    "Standard OT_time": time.time() - t0,
                    "Standard OT_knn_dist": calculate_pi_weighted_cost(plan_std, cost_knn, n_src),
                    "Standard OT_confidence": conf_std,
                    "Standard OT_wasserstein_dist": wass_std,
                    "Standard OT_sparsity": float('nan')
                })

            if "Causal OT" in baselines_to_run:
                t0 = time.time()
                plan_causal, cost_U = get_ot_plan_and_cost(U_rej, U_acc)
                cost_X = ((X_rej.unsqueeze(1) - X_acc.unsqueeze(0)) ** 2).sum(dim=2).numpy()
                conf_causal, wass_causal = compute_extra_metrics(plan_causal, X_acc, model, is_ot=True)
                
                # Proyección baricéntrica en el espacio causal y mapeo de vuelta para medir validez
                proj_U_causal = get_barycentric_proj(plan_causal, U_acc)
                proj_X_causal = proj_U_causal @ M_t.T
                validity_causal = (model(proj_X_causal) > 0).float().mean().item() * 100
                
                fold_res.update({
                    "Causal OT_validity_real": validity_causal, 
                    "Causal OT_ambient_cost": calculate_pi_weighted_cost(plan_causal, cost_X, n_src),
                    "Causal OT_latent_effort": calculate_pi_weighted_cost(plan_causal, cost_U, n_src),
                    "Causal OT_time": time.time() - t0,
                    "Causal OT_knn_dist": calculate_pi_weighted_cost(plan_causal, cost_knn, n_src),
                    "Causal OT_confidence": conf_causal,
                    "Causal OT_wasserstein_dist": wass_causal,
                    "Causal OT_sparsity": float('nan')
                })

            det_methods = [m for m in baselines_to_run if "OT" not in m]
            baselines = {}
            
            if "Wachter" in det_methods:
                t0 = time.time(); baselines["Wachter"] = {"proj_X": run_wachter(X_rej, model), "time": time.time() - t0}
            if "Karimi" in det_methods:
                t0 = time.time(); baselines["Karimi"] = {"proj_X": run_karimi(X_rej, M_t, M_inv_t, model), "time": time.time() - t0}
            if "FACEGroup" in det_methods:
                t0 = time.time(); baselines["FACEGroup"] = {"proj_X": run_facegroup(X_rej, X_acc), "time": time.time() - t0}
            if "Carrizosa_Exact" in det_methods:
                t0 = time.time(); baselines["Carrizosa_Exact"] = {"proj_X": run_gurobi_baselines(X_rej, M_inv_t, model, "carrizosa"), "time": time.time() - t0}
            if "Karimi_Hard" in det_methods:
                t0 = time.time(); baselines["Karimi_Hard"] = {"proj_X": run_gurobi_baselines(X_rej, M_inv_t, model, "karimi_hard"), "time": time.time() - t0}

            for method in det_methods:
                if method not in baselines: continue
                data = baselines[method]
                proj_X = data["proj_X"]
                proj_U = proj_X @ M_inv_t.T
                
                conf_det, wass_det = compute_extra_metrics(proj_X, X_acc, model, is_ot=False)
                
                with torch.no_grad():
                    fold_res.update({
                        f"{method}_validity_real": (model(proj_X) > 0).float().mean().item() * 100,
                        f"{method}_ambient_cost": ((X_rej - proj_X) ** 2).sum(dim=1).mean().item(),
                        f"{method}_latent_effort": ((U_rej - proj_U) ** 2).sum(dim=1).mean().item(),
                        f"{method}_time": data["time"],
                        f"{method}_knn_dist": compute_knn_distance(proj_X, X_acc),
                        f"{method}_confidence": conf_det,
                        f"{method}_wasserstein_dist": wass_det,
                        f"{method}_sparsity": len(torch.unique(proj_X, dim=0))
                    })
            all_rows.append(fold_res)

    df = pd.DataFrame(all_rows)
    path = Path(__file__).resolve().parents[3] / "results" / "experiment_carrizosa.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    
    if path.exists():
        existing_df = pd.read_csv(path)
        keys = ['dataset', 'fold']
        if all(k in existing_df.columns for k in keys) and all(k in df.columns for k in keys):
            existing_df.set_index(keys, inplace=True)
            df.set_index(keys, inplace=True)
            df = df.combine_first(existing_df).reset_index()
        else:
            df = pd.concat([existing_df, df], ignore_index=True)
            
    df.to_csv(path, index=False)
    print(f"\nResults saved at: {path}")
    return df

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=ALL_DATASETS, help="List of datasets to run")
    parser.add_argument("--baselines", nargs="+", default=ALL_BASELINES, help="List of baselines to run")
    args = parser.parse_args()
    
    run_unified_experiments(datasets_to_run=args.datasets, baselines_to_run=args.baselines)