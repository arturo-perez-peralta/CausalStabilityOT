from typing import Dict


DATASET_CONFIGS: Dict[str, Dict] = {
    "german": {
        "numerical_cols": [
            "age", "duration", "credit_amount",
            "installment_commitment", "residence_since",
            "existing_credits", "num_dependents",
        ],
        "categorical_cols": [
            "personal_status", "housing",
            "checking_status", "credit_history", "purpose",
            "savings_status", "employment", "other_parties",
            "property_magnitude", "other_payment_plans",
            "job", "own_telephone", "foreign_worker",
        ],
        "sensitive_single": ["sex"],
        "sensitive_intersectional": ["sex", "age_ge_25"],
    },
    "adult": {
        "numerical_cols": ["age", "education-num", "hours-per-week"],
        "categorical_cols": ["sex", "race"],
        "sensitive_single": ["sex"],
        "sensitive_intersectional": ["sex", "race"],
    },
    "compas": {
        "numerical_cols": [
            "age", "priors_count", "juv_fel_count",
            "juv_misd_count", "juv_other_count",
        ],
        "categorical_cols": ["race", "sex", "c_charge_degree"],
        "sensitive_single": ["race"],
        "sensitive_intersectional": ["race", "sex"],
    },
    "lsac": {
        "numerical_cols": ["age", "decile1", "decile3", "fam_inc", "lsat", "ugpa"],
        "categorical_cols": ["gender", "race1", "cluster", "fulltime"],
        "sensitive_single": ["race1"],
        "sensitive_intersectional": ["race1", "gender"],
    },
    "credit_default": {
        "numerical_cols": [
            "LIMIT_BAL", "AGE",
            "BILL_AMT1", "BILL_AMT2", "BILL_AMT3",
            "BILL_AMT4", "BILL_AMT5", "BILL_AMT6",
            "PAY_AMT1", "PAY_AMT2", "PAY_AMT3",
            "PAY_AMT4", "PAY_AMT5", "PAY_AMT6",
        ],
        "categorical_cols": [
            "SEX", "EDUCATION", "MARRIAGE",
            "PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6",
        ],
        "sensitive_single": ["SEX"],
        "sensitive_intersectional": ["SEX", "AGE_GE_25"],
    },
    "student": {
        "numerical_cols": [
            "age", "Medu", "Fedu", "traveltime",
            "studytime", "failures", "absences",
            "famrel", "freetime", "goout", "Dalc", "Walc", "health",
        ],
        "categorical_cols": [
            "sex", "address", "famsize", "Pstatus",
            "Mjob", "Fjob", "reason", "guardian",
            "schoolsup", "famsup", "paid", "activities",
            "nursery", "higher", "internet", "romantic",
            "school",
        ],
        "sensitive_single": ["sex"],
        "sensitive_intersectional": ["sex", "age_ge_17"],
    },
}


def get_config(dataset_name: str) -> Dict:
    name = dataset_name.lower().strip()
    if name not in DATASET_CONFIGS:
        raise ValueError(f"Unknown dataset '{name}'. Available: {list(DATASET_CONFIGS.keys())}")
    return DATASET_CONFIGS[name]
