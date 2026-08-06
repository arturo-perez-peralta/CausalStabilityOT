import torch
import pytest
from fairopt.utils.ilr import ilr_transform, inverse_ilr_transform


class TestILRTransform:
    def test_roundtrip(self):
        torch.manual_seed(42)
        simplex = torch.softmax(torch.randn(100, 5), dim=-1)
        z = ilr_transform(simplex)
        recovered = inverse_ilr_transform(z)
        assert torch.allclose(simplex, recovered, atol=1e-6), "ILR roundtrip failed"

    def test_output_shape(self):
        simplex = torch.softmax(torch.randn(50, 4), dim=-1)
        z = ilr_transform(simplex)
        assert z.shape[-1] == 3, f"Expected d-1=3, got {z.shape[-1]}"

    def test_inverse_output_shape(self):
        z = torch.randn(50, 3)
        simplex = inverse_ilr_transform(z)
        assert simplex.shape[-1] == 4, f"Expected d=4, got {simplex.shape[-1]}"

    def test_simplex_property(self):
        z = torch.randn(10, 4)
        simplex = inverse_ilr_transform(z)
        row_sums = simplex.sum(dim=-1)
        assert torch.allclose(row_sums, torch.ones(10), atol=1e-6)

    def test_nonnegative(self):
        z = torch.randn(10, 3)
        simplex = inverse_ilr_transform(z)
        assert (simplex >= 0).all(), "Simplex has negative entries"

    def test_batch_roundtrip(self):
        torch.manual_seed(0)
        for d in range(2, 8):
            simplex = torch.softmax(torch.randn(20, d), dim=-1)
            z = ilr_transform(simplex)
            recovered = inverse_ilr_transform(z)
            assert torch.allclose(simplex, recovered, atol=1e-5), f"Failed at d={d}"
