import torch
import pytest
from fairopt.core.sinkhorn import SinkhornSolver


class TestSinkhornSolver:
    def test_converges_on_toy_gaussians(self):
        torch.manual_seed(42)
        n, m = 50, 50
        mu0 = torch.randn(n, 2) + 2.0
        mu1 = torch.randn(m, 2)
        cost = torch.cdist(mu0, mu1, p=2)
        mu_n = torch.ones(n) / n
        nu_m = torch.ones(m) / m

        solver = SinkhornSolver(epsilon=0.1, max_iter=2000, tol=1e-3)
        result = solver.solve(mu_n, nu_m, cost)

        assert result.converged, "Sinkhorn did not converge"
        assert result.plan.shape == (n, m)
        assert result.f.shape == (n,)
        assert result.g.shape == (m,)
        assert result.marginal_error < 1e-2

    def test_marginal_constraints(self):
        torch.manual_seed(0)
        n, m = 30, 40
        mu0 = torch.randn(n, 2)
        mu1 = torch.randn(m, 2)
        cost = torch.cdist(mu0, mu1, p=2)
        mu_n = torch.ones(n) / n
        nu_m = torch.ones(m) / m

        solver = SinkhornSolver(epsilon=0.05, max_iter=500, tol=1e-9)
        result = solver.solve(mu_n, nu_m, cost)

        row_sum = result.plan.sum(dim=1)
        col_sum = result.plan.sum(dim=0)

        assert torch.allclose(row_sum, mu_n, atol=1e-4), "Row marginals violated"
        assert torch.allclose(col_sum, nu_m, atol=1e-4), "Column marginals violated"

    def test_epsilon_large_gives_uniform_plan(self):
        n, m = 10, 10
        mu0 = torch.randn(n, 2)
        mu1 = torch.randn(m, 2)
        cost = torch.cdist(mu0, mu1, p=2)
        mu_n = torch.ones(n) / n
        nu_m = torch.ones(m) / m

        solver = SinkhornSolver(epsilon=100.0, max_iter=100, tol=1e-9)
        result = solver.solve(mu_n, nu_m, cost)

        expected_uniform = torch.ones(n, m) / (n * m)
        assert torch.allclose(result.plan, expected_uniform, atol=1e-2)

    @pytest.mark.parametrize("n,m", [(10, 10), (20, 5), (5, 20)])
    def test_various_shapes(self, n, m):
        mu0 = torch.randn(n, 2)
        mu1 = torch.randn(m, 2)
        cost = torch.cdist(mu0, mu1, p=2)
        mu_n = torch.ones(n) / n
        nu_m = torch.ones(m) / m

        solver = SinkhornSolver(epsilon=0.1, max_iter=2000, tol=1e-3)
        result = solver.solve(mu_n, nu_m, cost)

        assert result.plan.shape == (n, m)
        assert result.converged

    def test_plan_nonnegative(self):
        n, m = 20, 30
        mu0 = torch.randn(n, 2)
        mu1 = torch.randn(m, 2)
        cost = torch.cdist(mu0, mu1, p=2)
        mu_n = torch.ones(n) / n
        nu_m = torch.ones(m) / m

        solver = SinkhornSolver(epsilon=0.1, max_iter=500, tol=1e-9)
        result = solver.solve(mu_n, nu_m, cost)

        assert (result.plan >= 0).all(), "Plan has negative entries"
