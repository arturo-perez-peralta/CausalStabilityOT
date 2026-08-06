"""
C1: Causal optimal transport stability under anisotropic deformations.

SCM: X = M U, where M = I + c N with N strictly lower triangular.
Source noise nu_U^0: uniform on the unit ball.
Target noise nu_U^1: pushforward of nu_U^0 by P(u) = S u + t.
Ambient distributions: mu_X^0 = M_# nu_U^0, mu_X^1 = M_# nu_U^1.
Tracks conjugacy error as a function of causal strength c.
"""
from __future__ import annotations

import torch
import numpy as np
import pandas as pd
from pathlib import Path

from fairopt.core.cost import pairwise_l2
from fairopt.core.metrics import barycentric_projection
from fairopt.core.sinkhorn import SinkhornSolver

RESULTS_DIR = Path(__file__).resolve().parents[2] / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
LOG = RESULTS_DIR / "c1_run.log"


def log(msg: str) -> None:
    print(msg, flush=True)
    with open(LOG, "a") as f:
        f.write(msg + "\n")
        f.flush()


def generate_causal_N(d: int, rng: torch.Generator) -> torch.Tensor:
    N = torch.randn(d, d, generator=rng)
    return torch.tril(N, diagonal=-1)


def build_M(c: float, N: torch.Tensor) -> torch.Tensor:
    return torch.eye(N.shape[0]) + c * N


def compute_anisotropy(M: torch.Tensor) -> tuple:
    G = M.T @ M
    evals = torch.linalg.eigvalsh(G)
    lambda_min = evals.min().item()
    lambda_max = evals.max().item()
    kappa = lambda_max / lambda_min
    return lambda_max, lambda_min, kappa


def sample_uniform_ball(
    n: int, d: int, scaling_matrix: torch.Tensor, rng: torch.Generator,
) -> torch.Tensor:
    v = torch.randn(n, d, generator=rng)
    u = v / v.norm(dim=1, keepdim=True)
    r = torch.rand(n, 1, generator=rng) ** (1.0 / d)
    return (r * u) @ scaling_matrix


def compute_transport_map(
    source: torch.Tensor,
    target: torch.Tensor,
    epsilon: float,
    max_iter: int,
    tol: float,
) -> torch.Tensor | None:
    n_src = source.shape[0]
    n_tgt = target.shape[0]
    mu = torch.ones(n_src) / n_src
    nu = torch.ones(n_tgt) / n_tgt
    cost = pairwise_l2(source, target)
    solver = SinkhornSolver(epsilon=epsilon, max_iter=max_iter, tol=tol)
    result = solver.solve(mu, nu, cost)
    if result.plan is None or not result.converged:
        return None
    return barycentric_projection(result.plan, target)


def compute_integral_diff(
    proj_X: torch.Tensor, proj_U: torch.Tensor, M: torch.Tensor,
) -> float:
    conjugated = proj_U @ M.T
    diff = proj_X - conjugated
    return (diff ** 2).sum(dim=1).mean().item()


def row_wise_mse(A: torch.Tensor, B: torch.Tensor) -> float:
    return torch.mean(torch.sum((A - B) ** 2, dim=1)).item()


def compute_theoretical_costs(M: torch.Tensor, S: torch.Tensor, t: torch.Tensor, d: int) -> tuple[float, float]:
    Sigma_U = torch.eye(d) / (d + 2)

    I_minus_S = torch.eye(d) - S
    var_U = torch.trace(I_minus_S @ I_minus_S @ Sigma_U).item()
    mean_U = torch.sum(t ** 2).item()
    C_TU = var_U + mean_U

    Sigma_1 = M @ Sigma_U @ M.T
    Sigma_2 = M @ S @ S @ Sigma_U @ M.T

    sqrt_Sigma_1 = torch.linalg.cholesky(Sigma_1)
    cross_cov = sqrt_Sigma_1.T @ Sigma_2 @ sqrt_Sigma_1
    S_cross = torch.linalg.svdvals(cross_cov)
    trace_term = torch.sum(torch.sqrt(S_cross)).item()

    var_X = torch.trace(Sigma_1).item() + torch.trace(Sigma_2).item() - 2 * trace_term
    M_t = M @ t
    mean_X = torch.sum(M_t ** 2).item()
    C_TX = var_X + mean_X

    return C_TX, C_TU


SINKHORN_EPS = 1.0
SINKHORN_MAX_ITER = 500
SINKHORN_TOL = 1e-6
C_GRID = [0.0, 0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0]
N_SIM = 50


def run_version(
    d: int,
    n_obs: int,
    S: torch.Tensor,
    t: torch.Tensor,
    label: str,
) -> pd.DataFrame:
    log(f"Running {label}: d={d}, n_obs={n_obs}, n_sim={N_SIM}")
    log(f"    c_grid={C_GRID}")
    log(f"    S=diag({[round(float(S[i,i]), 2) for i in range(d)]})")
    log(f"    t={t.tolist()}")

    base_rng = torch.Generator().manual_seed(42)
    N = generate_causal_N(d, base_rng)

    rows = []

    for c in C_GRID:
        M = build_M(c, N)
        Lambda, lambda_min, kappa = compute_anisotropy(M)

        C_TX, C_TU = compute_theoretical_costs(M, S, t, d)
        bound_multiplier = kappa - 1.0

        integrals = []
        u_discrepancies = []
        M_inv = torch.linalg.inv(M)
        for sim in range(N_SIM):
            sim_rng = torch.Generator().manual_seed(42 + sim)

            U_s = sample_uniform_ball(n_obs, d, torch.eye(d), sim_rng)
            X_s = U_s @ M.T

            U_t_raw = sample_uniform_ball(n_obs, d, torch.eye(d), sim_rng)
            U_t = U_t_raw @ S.T + t
            X_t = U_t @ M.T

            proj_U = compute_transport_map(
                U_s, U_t, SINKHORN_EPS, SINKHORN_MAX_ITER, SINKHORN_TOL,
            )
            proj_X = compute_transport_map(
                X_s, X_t, SINKHORN_EPS, SINKHORN_MAX_ITER, SINKHORN_TOL,
            )
            if proj_U is None or proj_X is None:
                continue

            integral = compute_integral_diff(proj_X, proj_U, M)
            integrals.append(integral)

            T_X_pulled_back = proj_X @ M_inv.T
            u_disc = row_wise_mse(proj_U, T_X_pulled_back)
            u_discrepancies.append(u_disc)

        if integrals:
            integral_mean = float(np.mean(integrals))
            integral_std = float(np.std(integrals, ddof=1))
            u_disc_mean = float(np.mean(u_discrepancies))
            u_disc_std = float(np.std(u_discrepancies, ddof=1))
        else:
            integral_mean = float("nan")
            integral_std = float("nan")
            u_disc_mean = float("nan")
            u_disc_std = float("nan")

        rows.append({
            "c": c,
            "Lambda": Lambda,
            "lambda_min": lambda_min,
            "kappa": kappa,
            "C_TX": C_TX,
            "C_TU": C_TU,
            "bound_X": C_TX * bound_multiplier,
            "bound_U": C_TU * bound_multiplier,
            "integral_mean": integral_mean,
            "integral_std": integral_std,
            "u_discrepancy_mean": u_disc_mean,
            "u_discrepancy_std": u_disc_std,
        })

        log(
            f"  c={c:.3f}  Lambda={Lambda:.4f}  "
            f"lambda_min={lambda_min:.4f}  kappa={kappa:.4f}  "
            f"integral={integral_mean:.6f} +/- {integral_std:.6f}  "
            f"u_disc={u_disc_mean:.6f} +/- {u_disc_std:.6f}"
        )

    df = pd.DataFrame(rows)
    csv_name = f"c1_causality_{label}.csv"
    path = RESULTS_DIR / csv_name
    df.to_csv(path, index=False)
    log(f"Results saved to {path}")
    return df


def run() -> None:
    log("#" * 60)
    log("C1: Causal OT Stability Under Anisotropic Deformations")
    log("#" * 60)

    run_version(
        d=2, n_obs=3000,
        S=torch.diag(torch.tensor([1.0, 2.0])),
        t=torch.tensor([2.0, 0.0]),
        label="2d",
    )

    run_version(
        d=5, n_obs=5000,
        S=torch.diag(torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])),
        t=torch.tensor([2.0, 0.0, 0.0, 0.0, 0.0]),
        label="5d",
    )

    log("Done.")


if __name__ == "__main__":
    run()
