import torch


def pairwise_l2(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    X_norm = (X ** 2).sum(dim=1, keepdim=True)
    Y_norm = (Y ** 2).sum(dim=1, keepdim=True)
    dist = X_norm + Y_norm.T - 2.0 * (X @ Y.T)
    dist = dist.clamp(min=0)
    return dist.sqrt()


def pairwise_l1(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    return torch.cdist(X, Y, p=1)
