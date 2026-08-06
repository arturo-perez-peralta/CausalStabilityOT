import pandas as pd
import numpy as np
from typing import Tuple, Dict
from sklearn.datasets import fetch_openml


def load_german_credit() -> Tuple[pd.DataFrame, pd.Series, Dict]:
    data = fetch_openml(data_id=31, as_frame=True, parser="pandas")
    df = data.data
    target = (data.target.astype(str).str.strip().str.lower() == "good").astype(int)

    df["sex"] = df["personal_status"].astype(str).str.startswith("male").astype(int)
    df["age_ge_25"] = (df["age"] >= 25).astype(int)

    info = {
        "name": "German Credit",
        "n_samples": df.shape[0],
        "n_features": df.shape[1],
        "sensitive_attributes": ["age", "sex"],
    }
    return df, target, info


def load_adult() -> Tuple[pd.DataFrame, pd.Series, Dict]:
    data = fetch_openml(data_id=1590, as_frame=True, parser="pandas")
    df = data.data
    target = (data.target.astype(str).str.strip() == ">50K").astype(int)
    info = {
        "name": "Adult",
        "n_samples": df.shape[0],
        "n_features": df.shape[1],
        "sensitive_attributes": ["sex", "race"],
    }
    return df, target, info


def load_compas() -> Tuple[pd.DataFrame, pd.Series, Dict]:
    url = (
        "https://raw.githubusercontent.com/propublica/compas-analysis/"
        "master/compas-scores-two-years.csv"
    )
    df = pd.read_csv(url)
    df = df[df.days_b_screening_arrest <= 30]
    df = df[df.days_b_screening_arrest >= -30]
    df = df[df.is_recid != -1]
    df = df[df.c_charge_degree != "O"]
    df = df[df.score_text != "N/A"]

    df["two_year_recid"] = df["two_year_recid"].astype(int)
    target = df["two_year_recid"]

    feature_cols = [
        "age",
        "sex",
        "race",
        "priors_count",
        "juv_fel_count",
        "juv_misd_count",
        "juv_other_count",
        "c_charge_degree",
    ]
    df = df[feature_cols].copy()
    df["sex"] = (df["sex"] == "Male").astype(int)
    df["c_charge_degree"] = (df["c_charge_degree"] == "F").astype(int)

    info = {
        "name": "COMPAS",
        "n_samples": df.shape[0],
        "n_features": df.shape[1],
        "sensitive_attributes": ["race"],
    }
    return df, target, info


def load_lsac() -> Tuple[pd.DataFrame, pd.Series, Dict]:
    data = fetch_openml(data_id=43889, as_frame=True, parser="pandas")
    df = data.data
    target_col = None
    for col in df.columns:
        if col.lower() == "bar":
            target_col = col
            break
    if target_col:
        target = (df[target_col].astype(str).str.strip().str.upper() == "TRUE").astype(int)
        df = df.drop(columns=[target_col])
    else:
        target = pd.Series(np.zeros(df.shape[0]))

    info = {
        "name": "LSAC",
        "n_samples": df.shape[0],
        "n_features": df.shape[1],
        "sensitive_attributes": ["race", "gender"],
    }
    return df, target, info


def load_credit_default() -> Tuple[pd.DataFrame, pd.Series, Dict]:
    data = fetch_openml(data_id=42477, as_frame=True, parser="pandas")
    raw = data.data
    target = data.target.astype(int)
    col_map = {
        "x1": "LIMIT_BAL", "x2": "SEX", "x3": "EDUCATION", "x4": "MARRIAGE",
        "x5": "AGE",
        "x6": "PAY_0", "x7": "PAY_2", "x8": "PAY_3",
        "x9": "PAY_4", "x10": "PAY_5", "x11": "PAY_6",
        "x12": "BILL_AMT1", "x13": "BILL_AMT2", "x14": "BILL_AMT3",
        "x15": "BILL_AMT4", "x16": "BILL_AMT5", "x17": "BILL_AMT6",
        "x18": "PAY_AMT1", "x19": "PAY_AMT2", "x20": "PAY_AMT3",
        "x21": "PAY_AMT4", "x22": "PAY_AMT5", "x23": "PAY_AMT6",
    }
    df = raw.rename(columns=col_map)
    for c in ["SEX", "EDUCATION", "MARRIAGE"] + [f"PAY_{i}" for i in [0, 2, 3, 4, 5, 6]]:
        df[c] = df[c].astype(int)
    df["AGE_GE_25"] = (df["AGE"] >= 25).astype(int)
    info = {
        "name": "Credit Default",
        "n_samples": df.shape[0],
        "n_features": df.shape[1],
        "sensitive_attributes": ["SEX", "AGE"],
    }
    return df, target, info


def load_student() -> Tuple[pd.DataFrame, pd.Series, Dict]:
    data = fetch_openml(data_id=42351, as_frame=True, parser="pandas")
    df = data.data
    target = (data.target.astype(int) >= 10).astype(int)
    df = df.drop(columns=["G1", "G2"], errors="ignore")
    df["age_ge_17"] = (df["age"] >= 17).astype(int)
    info = {
        "name": "Student",
        "n_samples": df.shape[0],
        "n_features": df.shape[1],
        "sensitive_attributes": ["sex"],
    }
    return df, target, info


DATASET_REGISTRY = {
    "german": load_german_credit,
    "adult": load_adult,
    "compas": load_compas,
    "lsac": load_lsac,
    "credit_default": load_credit_default,
    "student": load_student,
}


def load_dataset(name: str) -> Tuple[pd.DataFrame, pd.Series, Dict]:
    name = name.lower().strip()
    if name not in DATASET_REGISTRY:
        raise ValueError(
            f"Unknown dataset '{name}'. Available: {list(DATASET_REGISTRY.keys())}"
        )
    return DATASET_REGISTRY[name]()
