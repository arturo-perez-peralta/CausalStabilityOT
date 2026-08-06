import torch
import time
from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class MetricsResult:
    avg_transport_cost: float
    execution_time: float
    individual_variance: float
    per_individual_variance: float
    group_variances: Dict[str, float]


def avg_transport_cost(plan: torch.Tensor, cost_matrix: torch.Tensor) -> float:
    total_cost = (plan * cost_matrix).sum()
    total_mass = plan.sum()
    return (total_cost / total_mass).item() if total_mass > 0 else 0.0


def individual_variance(plan: torch.Tensor, cost_matrix: torch.Tensor) -> float:
    total_mass = plan.sum()
    mean_cost = (plan * cost_matrix).sum() / total_mass
    var = (plan * ((cost_matrix - mean_cost) ** 2)).sum() / total_mass
    return var.item()


def per_individual_variance(plan: torch.Tensor, cost_matrix: torch.Tensor) -> float:
    mass_i = plan.sum(dim=1)
    expected_cost_i = (plan * cost_matrix).sum(dim=1) / mass_i.clamp(min=1e-12)
    valid = mass_i > 1e-12
    if valid.sum() <= 1:
        return 0.0
    mean = expected_cost_i[valid].mean()
    var = ((expected_cost_i[valid] - mean) ** 2).mean()
    return var.item()


def group_variance(
    plan: torch.Tensor, cost_matrix: torch.Tensor, group_indices: Dict[str, List[int]]
) -> Dict[str, float]:
    variances = {}
    for group_name, indices in group_indices.items():
        plan_g = plan[indices, :]
        cost_g = cost_matrix[indices, :]
        total_mass_g = plan_g.sum()
        if total_mass_g > 0:
            mean_cost_g = (plan_g * cost_g).sum() / total_mass_g
            var_g = (plan_g * ((cost_g - mean_cost_g) ** 2)).sum() / total_mass_g
            variances[group_name] = var_g.item()
        else:
            variances[group_name] = 0.0
    return variances


def barycentric_projection(plan: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    row_sum = plan.sum(dim=1, keepdim=True).clamp(min=1e-12)
    return (plan @ target) / row_sum


def compute_barycentric_w2(
    projected_source: torch.Tensor,
    target: torch.Tensor,
    epsilon: float = 0.1,
    max_iter: int = 100,
    tol: float = 1e-4,
    device: Optional[str] = None,
) -> float:
    from fairopt.core.sinkhorn import SinkhornSolver
    from fairopt.core.cost import pairwise_l2

    n0 = projected_source.shape[0]
    n1 = target.shape[0]
    cost = pairwise_l2(projected_source, target)
    dev = device or projected_source.device
    mu = torch.ones(n0, device=dev) / n0
    nu = torch.ones(n1, device=dev) / n1
    result = SinkhornSolver(epsilon=epsilon, max_iter=max_iter, tol=tol, device=dev).solve(mu, nu, cost)
    if result.plan is None:
        return 0.0
    return avg_transport_cost(result.plan, cost)


def per_group_barycentric_w2_sum(
    plan: torch.Tensor,
    target: torch.Tensor,
    group_indices: dict,
    epsilon: float = 0.1,
    max_iter: int = 100,
    tol: float = 1e-4,
    device: Optional[str] = None,
) -> float:
    total = 0.0
    for name, idx in group_indices.items():
        plan_g = plan[idx]
        projected = barycentric_projection(plan_g, target)
        w2 = compute_barycentric_w2(projected, target, epsilon, max_iter, tol, device)
        total += w2
    return total


class Timer:
    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.elapsed = time.perf_counter() - self.start
