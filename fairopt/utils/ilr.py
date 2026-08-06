import torch


def _build_ilr_matrix(d: int, device: torch.device = torch.device("cpu")) -> torch.Tensor:
    M = torch.zeros(d, d - 1, device=device)
    for i in range(d - 1):
        r = d - i - 1
        M[i, i] = torch.sqrt(torch.tensor(r / (r + 1), device=device))
        M[i + 1 :, i] = -1.0 / torch.sqrt(torch.tensor(r * (r + 1), device=device))
    return M


def clr_transform(x: torch.Tensor) -> torch.Tensor:
    x = x.clamp(min=1e-12)
    geometric_mean = x.prod(dim=-1, keepdim=True).pow(1.0 / x.shape[-1])
    return torch.log(x / geometric_mean)


def inverse_clr_transform(y: torch.Tensor) -> torch.Tensor:
    return torch.softmax(y, dim=-1)


def ilr_transform(x: torch.Tensor) -> torch.Tensor:
    d = x.shape[-1]
    M = _build_ilr_matrix(d, device=x.device)
    y = clr_transform(x)
    return y @ M


def inverse_ilr_transform(z: torch.Tensor) -> torch.Tensor:
    d = z.shape[-1] + 1
    M = _build_ilr_matrix(d, device=z.device)
    y = z @ M.T
    return inverse_clr_transform(y)
