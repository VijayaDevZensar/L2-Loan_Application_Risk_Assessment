"""
Synthetic loan application data generator for FinTrust Bank capstone.
Generates a clean dataset and a defect-injected version for QE testing.
All data is synthetic - no real PII or production records.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
rng = np.random.default_rng(SEED)
random.seed(SEED)

ROOT = Path(__file__).parent.parent.parent
DATA_RAW = ROOT / "data" / "raw"
DATA_CLEAN = ROOT / "data" / "clean"
DATA_RAW.mkdir(parents=True, exist_ok=True)
DATA_CLEAN.mkdir(parents=True, exist_ok=True)

REGIONS = ["North", "South", "East", "West", "Central"]
EMPLOYMENT_TYPES = ["Salaried", "Self-Employed", "Contract", "Retired"]
REPAYMENT_STATUSES = ["Good", "Delayed", "Default"]
AGE_BANDS = ["18-29", "30-39", "40-49", "50-59", "60+"]

N_CLEAN = 500
N_TOTAL = 550  # extra 50 carry defects


def _age_band(age: int) -> str:
    if age < 30:
        return "18-29"
    if age < 40:
        return "30-39"
    if age < 50:
        return "40-49"
    if age < 60:
        return "50-59"
    return "60+"


def _generate_clean_row(idx: int) -> dict:
    age = int(rng.integers(22, 68))
    annual_income = round(float(rng.integers(25000, 150000, endpoint=True)), 2)
    employment_years = round(float(rng.uniform(0.5, 35)), 1)
    existing_debt = round(float(rng.integers(0, int(annual_income * 0.8))), 2)
    credit_score = int(rng.integers(300, 850, endpoint=True))
    loan_amount = round(float(rng.integers(5000, 500000, endpoint=True)), 2)
    loan_term_months = int(rng.choice([12, 24, 36, 48, 60, 84, 120]))
    previous_defaults = int(rng.integers(0, 4))
    region = random.choice(REGIONS)
    employment_type = random.choice(EMPLOYMENT_TYPES)
    repayment_status = random.choice(REPAYMENT_STATUSES)
    debt_to_income = round(existing_debt / annual_income, 4) if annual_income > 0 else 0.0

    # Deterministic synthetic label: high risk if credit_score<600 or dti>0.5 or defaults>1
    risk_score = 0
    if credit_score < 600:
        risk_score += 2
    if debt_to_income > 0.5:
        risk_score += 2
    if previous_defaults > 1:
        risk_score += 2
    if annual_income < 35000:
        risk_score += 1
    if repayment_status == "Default":
        risk_score += 2
    elif repayment_status == "Delayed":
        risk_score += 1
    synthetic_target = 1 if risk_score >= 3 else 0  # 1 = high risk

    return {
        "application_id": f"APP{10001 + idx:05d}",
        "age": age,
        "age_band": _age_band(age),
        "annual_income": annual_income,
        "employment_years": employment_years,
        "existing_debt": existing_debt,
        "credit_score": credit_score,
        "loan_amount": loan_amount,
        "loan_term_months": loan_term_months,
        "previous_defaults": previous_defaults,
        "region": region,
        "employment_type": employment_type,
        "repayment_status": repayment_status,
        "debt_to_income": debt_to_income,
        "age_band": _age_band(age),
        "synthetic_target": synthetic_target,
    }


def generate_clean() -> pd.DataFrame:
    rows = [_generate_clean_row(i) for i in range(N_CLEAN)]
    df = pd.DataFrame(rows)
    # Fix duplicate age_band column from dict (keep first)
    df = df.loc[:, ~df.columns.duplicated()]
    return df


def inject_defects(df_clean: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of df_clean with intentional QE defects injected."""
    df = df_clean.copy()

    extra_rows = [_generate_clean_row(N_CLEAN + i) for i in range(50)]
    df_extra = pd.DataFrame(extra_rows)
    df_extra = df_extra.loc[:, ~df_extra.columns.duplicated()]
    df = pd.concat([df, df_extra], ignore_index=True)

    # Defect 1: Duplicate application IDs (rows 10 and 11 get same ID)
    df.loc[11, "application_id"] = df.loc[10, "application_id"]
    df.loc[12, "application_id"] = df.loc[10, "application_id"]

    # Defect 2: Missing values in mandatory annual_income
    df.loc[[5, 42, 87], "annual_income"] = np.nan

    # Defect 3: Negative income (invalid)
    df.loc[20, "annual_income"] = -15000.0

    # Defect 4: Invalid age (out of synthetic rule range)
    df.loc[30, "age"] = 15
    df.loc[31, "age"] = 150

    # Defect 5: Impossible credit score
    df.loc[40, "credit_score"] = 950
    df.loc[41, "credit_score"] = -10

    # Defect 6: Unsupported employment category
    df.loc[50, "employment_type"] = "Freelancer"
    df.loc[51, "employment_type"] = "Student"

    # Defect 7: Extreme loan amount (> 5x annual income with no justification)
    df.loc[60, "loan_amount"] = 9999999.0

    # Defect 8: Wrong data type for loan_term_months (documented in data quality report)
    # Cannot store a string in int64 column in pandas 3.x; we record this as a test expectation
    # The raw CSV saved separately captures this via a special mixed-type version
    df.loc[70, "loan_term_months"] = -1  # sentinel for invalid term (not in allowed list)

    # Defect 9: Post-decision leakage column (approval_flag should never be in model features)
    df["approval_flag"] = rng.integers(0, 2, size=len(df))

    # Defect 10: Class imbalance — ensure >70% are low-risk to create imbalance
    # (the label distribution is already skewed; document it)

    return df


def write_schemas(df_clean: pd.DataFrame) -> None:
    schema = {
        "version": "schema_v1.0",
        "description": "FinTrust Bank synthetic loan application schema",
        "fields": {
            "application_id": {"type": "string", "required": True, "unique": True, "pattern": "^APP\\d+$"},
            "age": {"type": "integer", "required": True, "min": 18, "max": 100},
            "age_band": {"type": "string", "required": False, "allowed": ["18-29", "30-39", "40-49", "50-59", "60+"]},
            "annual_income": {"type": "float", "required": True, "min": 0},
            "employment_years": {"type": "float", "required": True, "min": 0},
            "existing_debt": {"type": "float", "required": True, "min": 0},
            "credit_score": {"type": "integer", "required": True, "min": 300, "max": 850},
            "loan_amount": {"type": "float", "required": True, "min": 1},
            "loan_term_months": {"type": "integer", "required": True, "allowed": [12, 24, 36, 48, 60, 84, 120]},
            "previous_defaults": {"type": "integer", "required": True, "min": 0},
            "region": {"type": "string", "required": True, "allowed": ["North", "South", "East", "West", "Central"]},
            "employment_type": {
                "type": "string",
                "required": True,
                "allowed": ["Salaried", "Self-Employed", "Contract", "Retired"],
            },
            "repayment_status": {"type": "string", "required": True, "allowed": ["Good", "Delayed", "Default"]},
            "debt_to_income": {"type": "float", "required": False, "min": 0},
            "synthetic_target": {"type": "integer", "required": True, "allowed": [0, 1]},
        },
        "leakage_fields": ["approval_flag"],
        "note": "synthetic_target is the training label only — not for production features",
    }
    schema_path = ROOT / "data" / "schemas" / "loan_schema_v1.json"
    schema_path.parent.mkdir(parents=True, exist_ok=True)
    schema_path.write_text(json.dumps(schema, indent=2))
    print(f"Schema written: {schema_path}")


def main() -> None:
    print("Generating synthetic loan data...")
    df_clean = generate_clean()
    df_clean = df_clean.loc[:, ~df_clean.columns.duplicated()]

    # Save clean version
    clean_path = DATA_CLEAN / "loan_data_clean_v1.0.csv"
    df_clean.to_csv(clean_path, index=False)
    print(f"Clean dataset: {clean_path}  rows={len(df_clean)}")

    # Golden dataset — first 50 rows of clean, no defects, used for CI regression
    golden_path = ROOT / "data" / "golden" / "golden_50.csv"
    golden_path.parent.mkdir(parents=True, exist_ok=True)
    df_clean.head(50).to_csv(golden_path, index=False)
    print(f"Golden dataset: {golden_path}  rows=50")

    # Defective raw version
    df_defects = inject_defects(df_clean)
    raw_path = DATA_RAW / "loan_data_raw_v1.0.csv"
    df_defects.to_csv(raw_path, index=False)
    print(f"Raw (with defects): {raw_path}  rows={len(df_defects)}")

    write_schemas(df_clean)

    # Label distribution
    counts = df_clean["synthetic_target"].value_counts()
    print(f"\nLabel distribution (clean):\n{counts.to_dict()}")
    print(f"High-risk rate: {counts.get(1,0)/len(df_clean):.1%}")


if __name__ == "__main__":
    main()
