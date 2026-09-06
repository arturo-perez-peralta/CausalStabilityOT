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

# --- GLOBAL CONFIGURATIONS ---
BNS = ["ecoli70", "magic-niab", "magic-irri", "arth150"]

REAL_DATASETS_CFG = {
    "german": {"chain_cols": ["age", "credit_amount", "duration"], "label": "German"},
    "compas": {"chain_cols": ["age", "juv_crimes", "priors_count"], "label": "COMPAS"},
    "adult": {"chain_cols": ["age", "education-num", "hours-per-week"], "label": "Adult"},
    "lsac": {"chain_cols": ["fam_inc", "ugpa", "lsat"], "label": "LSAC"},
}

ALL_DATASETS = BNS + list(REAL_DATASETS_CFG.keys())

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

def get_corrupted_M(B: np.ndarray, std_devs: np.ndarray, mode: str, eps: float, rng: np.random.RandomState):
    B_hat = B.copy()
    nonzero = B != 0
    if mode == "noise" and eps > 0: 
        B_hat[nonzero] += rng.normal(0, eps, B.shape)[nonzero]
    elif mode == "miss" and eps > 0:
        idx = np.argwhere(nonzero)
        n_drop = int(eps * len(idx))
        if n_drop > 0:
            for r, c in idx[rng.choice(len(idx), n_drop, replace=False)]: B_hat[r, c] = 0 
                
    try:
        M_hat = np.linalg.solve(np.eye(B.shape[0]) - B_hat, np.diag(std_devs))
        return torch.from_numpy(M_hat).float(), torch.from_numpy(np.linalg.inv(M_hat)).float(), B_hat
    except np.linalg.LinAlgError: return None, None, None

def get_ot_plan_and_cost(source: torch.Tensor, target: torch.Tensor):
    n_src, n_tgt = source.shape[0], target.shape[0]
    mu, nu = np.ones(n_src)/n_src, np.ones(n_tgt)/n_tgt
    cost = ((source.unsqueeze(1) - target.unsqueeze(0)) ** 2).sum(dim=2).numpy()
    plan = ot.emd(mu, nu, cost)
    return plan, cost

def calculate_pi_weighted_cost(plan: np.ndarray, cost: np.ndarray, n_src: int) -> float:
    expected_cost_per_instance = ((plan * n_src) * cost).sum(axis=1)
    return float(expected_cost_per_instance.mean())

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
            M_dist = ot.dist(proj_X_np, X_acc_np, metric='sqeuclidean')
            wass_dist = float(ot.emd2(a, b, M_dist))
        else:
            wass_dist = float('nan')
    else:
        X_cf = X_cf_or_plan
        with torch.no_grad():
            confidence = torch.sigmoid(model(X_cf)).mean().item() * 100
        
        X_cf_np = X_cf.cpu().detach().numpy()
        a, b = np.ones(len(X_cf_np)) / len(X_cf_np), np.ones(len(X_acc_np)) / len(X_acc_np)
        M_dist = ot.dist(X_cf_np, X_acc_np, metric='sqeuclidean')
        wass_dist = float(ot.emd2(a, b, M_dist))
        
    return confidence, wass_dist

def run_robust_karimi(X_rej, M_hat_t, M_hat_inv_t, model):
    U_rej = X_rej @ M_hat_inv_t.T
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
            current_X = current_U @ M_hat_t.T
            bce = torch.nn.functional.binary_cross_entropy_with_logits(model(current_X), target, reduction='none')
            loss = (mse + lambdas * bce).mean()
            loss.backward()
            optimizer.step()
            
        with torch.no_grad():
            current_X = (U_cf + delta) @ M_hat_t.T
            preds = model(current_X) > 0
            new_success = preds & ~is_successful
            final_X_cf[new_success] = current_X[new_success].detach()
            is_successful = is_successful | preds
            
            if is_successful.all(): break
            
            lambdas[~is_successful] *= 10.0
            U_cf = (U_cf + delta).detach()
            
    final_X_cf[~is_successful] = current_X[~is_successful].detach()
    return final_X_cf

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


def run_robust_karimi_hard(X_rej, M_hat_inv_t, B_hat, model):
    X_rej_np, U_rej_np = X_rej.numpy(), (X_rej @ M_hat_inv_t.T).numpy()
    d = X_rej_np.shape[1]
    
    W1, b1 = model.net[0].weight.data.numpy(), model.net[0].bias.data.numpy()
    W2, b2 = model.net[2].weight.data.numpy(), model.net[2].bias.data.numpy()
    W3, b3 = model.net[4].weight.data.numpy(), model.net[4].bias.data.numpy()
    
    N_JOBS = 8
    res = Parallel(n_jobs=N_JOBS)(delayed(solve_single_karimi_hard)(X_rej_np[i], U_rej_np[i], B_hat, d, W1, b1, W2, b2, W3, b3) for i in range(X_rej.shape[0]))
    return torch.stack(res)

def run_robustness_experiments(datasets_to_run=None, baselines_to_run=None, n_obs: int = 3000) -> pd.DataFrame:
    datasets_to_run = datasets_to_run or BNS
    baselines_to_run = baselines_to_run or ["Standard OT", "Causal OT", "Karimi", "Karimi_Hard"]
    
    eps_noise = [1.5]
    eps_miss = []
    all_rows = []

    for bn_name in datasets_to_run:
        node_names, B, intercepts, std_devs, M, terminal_idx = load_bn(bn_name)
        d_bn, M_t = M.shape[0], torch.from_numpy(M).float()
        M_inv_t = torch.linalg.inv(M_t)

        rng = np.random.RandomState(42)
        X = torch.randn(n_obs, d_bn) @ M_t.T
        y = (X[:, terminal_idx] > X[:, terminal_idx].median()).numpy().astype(float)

        for fold, (train_idx, val_idx) in enumerate(KFold(n_splits=5, shuffle=True, random_state=42).split(X.numpy())):
            X_train, X_val, y_train, y_val = X[train_idx], X[val_idx], torch.from_numpy(y[train_idx]).float(), torch.from_numpy(y[val_idx]).float()

            model = MLP(d_bn)
            optimizer, criterion = optim.Adam(model.parameters(), lr=0.01), nn.BCEWithLogitsLoss()
            for _ in range(100):
                optimizer.zero_grad()
                loss = criterion(model(X_train), y_train)
                loss.backward(); optimizer.step()
                
            model.eval()
            with torch.no_grad(): preds_val = (model(X_val) > 0).float()
            
            X_rej, X_acc = X_val[preds_val == 0], X_val[preds_val == 1]
            if X_rej.shape[0] < 5 or X_acc.shape[0] < 5: continue
            
            n_src = X_rej.shape[0]
            U_rej_true = X_rej @ M_inv_t.T
            U_acc_true = X_acc @ M_inv_t.T
            
            configs = [{"mode": "noise", "eps": e} for e in eps_noise] + [{"mode": "miss", "eps": e} for e in eps_miss]

            for conf in configs:
                M_hat_t, M_hat_inv_t, B_hat = get_corrupted_M(B, std_devs, conf["mode"], conf["eps"], rng)
                if M_hat_t is None: continue
                    
                row = {"dataset": bn_name, "fold": fold, "error_mode": conf["mode"], "epsilon": conf["eps"]}

                cost_X_true = ((X_rej.unsqueeze(1) - X_acc.unsqueeze(0)) ** 2).sum(dim=2).numpy()
                cost_U_true = ((U_rej_true.unsqueeze(1) - U_acc_true.unsqueeze(0)) ** 2).sum(dim=2).numpy()

                if "Standard OT" in baselines_to_run:
                    try:
                        plan_std, _ = get_ot_plan_and_cost(X_rej, X_acc)
                        conf_std, wass_std = compute_extra_metrics(plan_std, X_acc, model, is_ot=True)
                        row["Standard OT_validity_real"] = 100.0
                        row["Standard OT_ambient_cost"] = calculate_pi_weighted_cost(plan_std, cost_X_true, n_src)
                        row["Standard OT_latent_effort"] = calculate_pi_weighted_cost(plan_std, cost_U_true, n_src)
                        row["Standard OT_confidence"] = conf_std
                        row["Standard OT_wasserstein_dist"] = wass_std
                    except Exception as e: pass
                
                if "Causal OT" in baselines_to_run:
                    try:
                        U_rej_hat, U_acc_hat = X_rej @ M_hat_inv_t.T, X_acc @ M_hat_inv_t.T
                        plan_causal, _ = get_ot_plan_and_cost(U_rej_hat, U_acc_hat)
                        conf_causal, wass_causal = compute_extra_metrics(plan_causal, X_acc, model, is_ot=True)
                        row["Causal OT_validity_real"] = 100.0
                        row["Causal OT_ambient_cost"] = calculate_pi_weighted_cost(plan_causal, cost_X_true, n_src)
                        row["Causal OT_latent_effort"] = calculate_pi_weighted_cost(plan_causal, cost_U_true, n_src)
                        row["Causal OT_confidence"] = conf_causal
                        row["Causal OT_wasserstein_dist"] = wass_causal
                    except Exception as e: pass

                methods = {}
                if "Karimi" in baselines_to_run: methods["Karimi"] = lambda: run_robust_karimi(X_rej, M_hat_t, M_hat_inv_t, model)
                if "Karimi_Hard" in baselines_to_run: methods["Karimi_Hard"] = lambda: run_robust_karimi_hard(X_rej, M_hat_inv_t, B_hat, model)

                for m_name, func in methods.items():
                    try:
                        X_cf = func()
                        conf_det, wass_det = compute_extra_metrics(X_cf, X_acc, model, is_ot=False)
                        row[f"{m_name}_validity_real"] = (model(X_cf) > 0).float().mean().item() * 100
                        row[f"{m_name}_ambient_cost"] = ((X_rej - X_cf) ** 2).sum(dim=1).mean().item()
                        row[f"{m_name}_latent_effort"] = ((U_rej_true - (X_cf @ M_inv_t.T)) ** 2).sum(dim=1).mean().item()
                        row[f"{m_name}_confidence"] = conf_det
                        row[f"{m_name}_wasserstein_dist"] = wass_det
                    except Exception as e:
                        print(f"Error {m_name}: {e}")
                        row[f"{m_name}_validity_real"] = row[f"{m_name}_ambient_cost"] = row[f"{m_name}_latent_effort"] = float('nan')
                        row[f"{m_name}_confidence"] = row[f"{m_name}_wasserstein_dist"] = float('nan')

                all_rows.append(row)
                print(f"[{bn_name} - {conf['mode']} eps={conf['eps']}] Causal OT Latent Effort: {row.get('Causal OT_latent_effort', np.nan):.2f}")

    df = pd.DataFrame(all_rows)
    path = Path(__file__).resolve().parents[3] / "results" / "experiment_bn_robustness.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    
    # Sobrescritura selectiva
    if path.exists():
        existing_df = pd.read_csv(path)
        keys = ['dataset', 'fold', 'error_mode', 'epsilon']
        if all(k in existing_df.columns for k in keys) and all(k in df.columns for k in keys):
            existing_df.set_index(keys, inplace=True)
            df.set_index(keys, inplace=True)
            df = df.combine_first(existing_df).reset_index()
        else:
            df = pd.concat([existing_df, df], ignore_index=True)
            
    df.to_csv(path, index=False)
    return df

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=BNS, help="List of datasets to run")
    parser.add_argument("--baselines", nargs="+", default=["Standard OT", "Causal OT", "Karimi", "Karimi_Hard"], help="List of baselines to run")
    args = parser.parse_args()
    
    run_robustness_experiments(datasets_to_run=args.datasets, baselines_to_run=args.baselines)