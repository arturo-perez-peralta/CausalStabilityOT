import torch
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler as SkStandardScaler
from typing import List, Optional, Tuple


class StandardScaler:
    def __init__(self):
        self.scaler = SkStandardScaler()

    def fit(self, X: torch.Tensor) -> None:
        self.scaler.fit(X.numpy())

    def transform(self, X: torch.Tensor) -> torch.Tensor:
        return torch.from_numpy(self.scaler.transform(X.numpy())).float()

    def fit_transform(self, X: torch.Tensor) -> torch.Tensor:
        return torch.from_numpy(self.scaler.fit_transform(X.numpy())).float()


def one_hot_encode(
    df: pd.DataFrame, columns: List[str]
) -> Tuple[pd.DataFrame, List[str]]:
    return pd.get_dummies(df, columns=columns, drop_first=False), [
        c for col in columns for c in pd.get_dummies(df[col], drop_first=False).columns
    ]


def train_test_split(
    *tensors: torch.Tensor, train_frac: float = 0.8, random_seed: int = 42
) -> List[torch.Tensor]:
    n = tensors[0].shape[0]
    rng = torch.Generator().manual_seed(random_seed)
    perm = torch.randperm(n, generator=rng)
    split = int(n * train_frac)
    train_idx = perm[:split]
    test_idx = perm[split:]
    result = []
    for t in tensors:
        result.append(t[train_idx])
        result.append(t[test_idx])
    return result


def to_tensor(df: pd.DataFrame) -> torch.Tensor:
    return torch.from_numpy(df.values.astype(np.float32)).float()


def categorical_to_tensor(series: pd.Series) -> torch.Tensor:
    codes = series.astype("category").cat.codes.values
    return torch.from_numpy(codes).long()
