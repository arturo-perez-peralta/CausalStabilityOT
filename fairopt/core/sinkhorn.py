import os
import torch
from dataclasses import dataclass
from typing import Optional

# Respect an explicit thread budget (FAIROPT_NUM_THREADS), otherwise keep the
# conservative 4-thread cap so we don't oversubscribe shared machines.
_n_threads = int(os.environ.get("FAIROPT_NUM_THREADS", "4"))
torch.set_num_threads(min(_n_threads, torch.get_num_threads()))


@dataclass
class SinkhornResult:
    plan: torch.Tensor
    f: torch.Tensor
    g: torch.Tensor
    u: torch.Tensor
    v: torch.Tensor
    marginal_error: float
    n_iter: int
    converged: bool


def _sinkhorn_loop(
    K: torch.Tensor,
    mu_n: torch.Tensor,
    nu_m: torch.Tensor,
    epsilon: float,
    max_iter: int,
    tol: float,
    check_every: int = 50,
):
    n = mu_n.shape[0]
    m = nu_m.shape[0]
    dev = K.device
    u = torch.ones(n, device=dev)
    v = torch.ones(m, device=dev)
    marginal_error = 1.0
    converged = False
    i = 0

    for i in range(max_iter):
        v_prev = v
        u_prev = u
        v = nu_m / (K.T @ u + 1e-12)
        u = mu_n / (K @ v + 1e-12)

        if i % check_every == 0:
            err_u = torch.norm(u - u_prev, p=float("inf")) / max(
                torch.norm(u, p=float("inf")), 1e-12
            )
            err_v = torch.norm(v - v_prev, p=float("inf")) / max(
                torch.norm(v, p=float("inf")), 1e-12
            )
            marginal_error = max(err_u.item(), err_v.item())
            if marginal_error < tol:
                converged = True
                break

    plan = torch.diag(u) @ K @ torch.diag(v)
    f = epsilon * torch.log(u.clamp(min=1e-12))
    g = epsilon * torch.log(v.clamp(min=1e-12))
    plan = plan.clamp(min=0)
    return plan, f, g, u, v, marginal_error, i + 1, converged


class SinkhornSolver:
    def __init__(
        self,
        epsilon: float = 0.1,
        max_iter: int = 1000,
        tol: float = 1e-9,
        device: Optional[torch.device] = None,
    ):
        self.epsilon = epsilon
        self.max_iter = max_iter
        self.tol = tol
        self.device = device or torch.device("cpu")

    def solve(
        self,
        mu_n: torch.Tensor,
        nu_m: torch.Tensor,
        cost_matrix: torch.Tensor,
    ) -> SinkhornResult:
        cost_max = cost_matrix.max().item()
        effective_eps = max(self.epsilon, cost_max / 50.0)

        K = torch.exp(-cost_matrix / effective_eps)
        K = K.to(self.device)
        mu_n = mu_n.to(self.device).reshape(-1)
        nu_m = nu_m.to(self.device).reshape(-1)

        plan, f, g, u, v, marginal_error, n_iter, converged = _sinkhorn_loop(
            K, mu_n, nu_m, effective_eps, self.max_iter, self.tol
        )

        return SinkhornResult(
            plan=plan,
            f=f,
            g=g,
            u=u,
            v=v,
            marginal_error=marginal_error,
            n_iter=n_iter,
            converged=converged,
        )
