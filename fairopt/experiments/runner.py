"""Shared data preparation for the causal experiments (C1-C3).

Only the helpers used by the causal experiments live here: ``prepare_data``,
``save_results`` and the ``Timer`` re-export.  The full CV harness used by the
fairness experiments lives in the ``fairness`` project.
"""
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from typing import Dict, List, Optional

from fairopt.core.metrics import Timer
from fairopt.data.datasets import load_dataset
from fairopt.data.preprocessing import StandardScaler


RESULTS_DIR = Path(__file__).resolve().parents[2] / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


_DATASET_MAX_SAMPLES = {
    "acsincome": 30000,
    "hmda": 30000,
    "adult": 30000,
}


def prepare_data(
    dataset_name: str,
    numerical_cols: List[str],
    categorical_cols: Optional[List[str]] = None,
    sensitive_cols: Optional[List[str]] = None,
    max_samples: Optional[int] = None,
) -> Dict:
    df, target, info = load_dataset(dataset_name)
    n = df.shape[0]

    _ds_max = _DATASET_MAX_SAMPLES.get(dataset_name.lower())
    if _ds_max is not None:
        max_samples = min(_ds_max, n)
    elif max_samples is None:
        max_samples = n  # use all observations

    if n > max_samples:
        rng_ss = np.random.RandomState(42)
        keep = rng_ss.choice(n, max_samples, replace=False)
        df = df.iloc[keep].reset_index(drop=True)
        target = target.iloc[keep].reset_index(drop=True)
        n = max_samples

    groups = None
    if sensitive_cols:
        groups = {}
        available = [c for c in sensitive_cols if c in df.columns]
        if len(available) == 1:
            col = available[0]
            for val in df[col].unique():
                mask = (df[col] == val).values
                groups[f"{col}_{val}"] = torch.from_numpy(mask.nonzero()[0].astype(np.int64))
        else:
            combined = df[available[0]].astype(str)
            for col in available[1:]:
                combined = combined + "_" + df[col].astype(str)
            for val in combined.unique():
                mask = (combined == val).values
                groups[str(val).replace(" ", "_")] = torch.from_numpy(mask.nonzero()[0].astype(np.int64))

    available_num = [c for c in numerical_cols if c in df.columns]
    X_num = df[available_num].values.astype(np.float32)
    scaler = StandardScaler()
    X_num_tensor = scaler.fit_transform(torch.from_numpy(X_num))

    X_cat_tensor = None
    if categorical_cols:
        available_cat = [c for c in categorical_cols if c in df.columns]
        if available_cat:
            X_cat_raw = df[available_cat].values
            X_cat_encoded = np.zeros((X_cat_raw.shape[0], len(available_cat)), dtype=np.int64)
            for i, col in enumerate(available_cat):
                col_data = X_cat_raw[:, i]
                codes, _ = pd.factorize(col_data, use_na_sentinel=False)
                X_cat_encoded[:, i] = codes
            X_cat_tensor = torch.from_numpy(X_cat_encoded)

    n = X_num_tensor.shape[0]
    rng = torch.Generator().manual_seed(42)
    perm = torch.randperm(n, generator=rng)
    split = int(n * 0.8)
    train_idx = perm[:split]
    test_idx = perm[split:]

    target_tensor = torch.from_numpy(target.values.astype(np.float32))

    result = {
        "X": X_num_tensor,
        "y": target_tensor,
        "train_idx": train_idx,
        "test_idx": test_idx,
        "groups": groups,
        "info": info,
    }
    if X_cat_tensor is not None:
        result["X_cat"] = X_cat_tensor

    return result


def save_results(df: pd.DataFrame, experiment_name: str) -> None:
    path = RESULTS_DIR / f"{experiment_name}.csv"
    df.to_csv(path, index=False)
    print(f"Results saved to {path}")
