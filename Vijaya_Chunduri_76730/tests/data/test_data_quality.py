"""
DATA QUALITY TESTS — Day 1, Exercise 2
FinTrust Bank AI QE Capstone

Covers ≥12 documented test cases, ≥5 automated validations:
DQ-01 through DQ-15
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd
import numpy as np
import pytest

ROOT = Path(__file__).parent.parent.parent
SCHEMA_PATH = ROOT / "data" / "schemas" / "loan_schema_v1.json"
CLEAN_PATH = ROOT / "data" / "clean" / "loan_data_clean_v1.0.csv"
RAW_PATH = ROOT / "data" / "raw" / "loan_data_raw_v1.0.csv"
GOLDEN_PATH = ROOT / "data" / "golden" / "golden_50.csv"

ALLOWED_EMPLOYMENT = ["Salaried", "Self-Employed", "Contract", "Retired"]
ALLOWED_REPAYMENT = ["Good", "Delayed", "Default"]
ALLOWED_REGIONS = ["North", "South", "East", "West", "Central"]
ALLOWED_TERMS = {12, 24, 36, 48, 60, 84, 120}


# ---- DQ-01: Schema conformity — all required fields present ----
def test_dq01_schema_required_fields_present(df_clean, schema):
    """DQ-01: Required fields must all be present in the clean dataset."""
    required = [col for col, meta in schema["fields"].items() if meta.get("required")]
    missing = [f for f in required if f not in df_clean.columns]
    assert missing == [], f"Missing required fields: {missing}"


# ---- DQ-02: Uniqueness — application_id must be unique ----
def test_dq02_application_id_unique_clean(df_clean):
    """DQ-02: application_id must be unique in the clean dataset."""
    dupes = df_clean["application_id"].duplicated().sum()
    assert dupes == 0, f"Found {dupes} duplicate application_ids in clean dataset"


def test_dq02_application_id_duplicate_detected_raw(df_raw):
    """DQ-02b: Duplicate application_ids ARE present in the raw dataset (known defect DFT-001)."""
    dupes = df_raw["application_id"].duplicated().sum()
    assert dupes > 0, "Expected duplicate application_ids in raw dataset — defect not injected?"


# ---- DQ-03: Completeness — annual_income not null ----
def test_dq03_annual_income_not_null_clean(df_clean):
    """DQ-03: annual_income must have zero nulls in the clean dataset (mandatory field)."""
    null_count = df_clean["annual_income"].isna().sum()
    assert null_count == 0, f"Clean dataset has {null_count} null annual_income values"


def test_dq03_missing_income_detected_raw(df_raw):
    """DQ-03b: Missing annual_income values ARE present in the raw dataset (known defect DFT-002)."""
    null_count = df_raw["annual_income"].isna().sum()
    assert null_count > 0, "Expected null annual_income in raw dataset — defect not injected?"


# ---- DQ-04: Validity — annual_income must be positive ----
def test_dq04_annual_income_positive_clean(df_clean):
    """DQ-04: annual_income must be > 0 in the clean dataset."""
    invalid = (df_clean["annual_income"] <= 0).sum()
    assert invalid == 0, f"Found {invalid} non-positive annual_income in clean dataset"


def test_dq04_negative_income_detected_raw(df_raw):
    """DQ-04b: Negative income IS present in the raw dataset (known defect DFT-003)."""
    invalid = (df_raw["annual_income"].dropna() <= 0).sum()
    assert invalid > 0, "Expected negative income in raw dataset — defect not injected?"


# ---- DQ-05: Validity — age within allowed range ----
def test_dq05_age_range_clean(df_clean):
    """DQ-05: age must be 18–100 in the clean dataset."""
    out_of_range = ((df_clean["age"] < 18) | (df_clean["age"] > 100)).sum()
    assert out_of_range == 0, f"Found {out_of_range} age values out of range [18,100]"


def test_dq05_invalid_age_detected_raw(df_raw):
    """DQ-05b: Invalid age values ARE present in raw dataset (known defect DFT-004)."""
    out_of_range = ((df_raw["age"] < 18) | (df_raw["age"] > 100)).sum()
    assert out_of_range > 0, "Expected invalid age in raw dataset — defect not injected?"


# ---- DQ-06: Validity — credit_score must be in [300, 850] ----
def test_dq06_credit_score_range_clean(df_clean):
    """DQ-06: credit_score must be in [300, 850]."""
    invalid = ((df_clean["credit_score"] < 300) | (df_clean["credit_score"] > 850)).sum()
    assert invalid == 0, f"Found {invalid} credit_score values out of [300, 850]"


def test_dq06_impossible_credit_score_detected_raw(df_raw):
    """DQ-06b: Impossible credit scores ARE present in raw dataset (known defect DFT-005)."""
    invalid = ((df_raw["credit_score"] < 300) | (df_raw["credit_score"] > 850)).sum()
    assert invalid > 0, "Expected impossible credit_score in raw dataset — defect not injected?"


# ---- DQ-07: Validity — employment_type uses allowed categories ----
def test_dq07_employment_type_categories_clean(df_clean):
    """DQ-07: employment_type must only contain allowed categories."""
    invalid = ~df_clean["employment_type"].isin(ALLOWED_EMPLOYMENT)
    assert invalid.sum() == 0, f"Unexpected employment_type values: {df_clean.loc[invalid, 'employment_type'].unique()}"


def test_dq07_unsupported_employment_detected_raw(df_raw):
    """DQ-07b: Unsupported employment categories ARE present in raw dataset (known defect DFT-006)."""
    invalid = ~df_raw["employment_type"].isin(ALLOWED_EMPLOYMENT)
    assert invalid.sum() > 0, "Expected unsupported employment_type in raw dataset — defect not injected?"


# ---- DQ-08: Validity — loan_term_months must be in approved set ----
def test_dq08_loan_term_allowed_values_clean(df_clean):
    """DQ-08: loan_term_months must be one of the approved values."""
    invalid = ~df_clean["loan_term_months"].isin(ALLOWED_TERMS)
    assert invalid.sum() == 0, f"Found invalid loan_term_months: {df_clean.loc[invalid, 'loan_term_months'].unique()}"


def test_dq08_invalid_term_detected_raw(df_raw):
    """DQ-08b: Invalid loan_term_months values (sentinel -1) ARE present in raw dataset (DFT-008)."""
    invalid_sentinel = (df_raw["loan_term_months"] == -1).sum()
    assert invalid_sentinel > 0, "Expected invalid loan_term_months in raw dataset — defect not injected?"


# ---- DQ-09: Validity — loan_amount must be positive ----
def test_dq09_loan_amount_positive_clean(df_clean):
    """DQ-09: loan_amount must be > 0."""
    invalid = (df_clean["loan_amount"] <= 0).sum()
    assert invalid == 0, f"Found {invalid} non-positive loan_amount values"


# ---- DQ-10: Leakage check — approval_flag must NOT be in clean dataset ----
def test_dq10_no_leakage_field_in_clean(df_clean, schema):
    """DQ-10: Post-decision leakage fields (approval_flag) must not appear in the clean training data."""
    leakage_fields = schema.get("leakage_fields", [])
    present = [f for f in leakage_fields if f in df_clean.columns]
    assert present == [], f"Leakage field(s) found in clean dataset: {present}"


def test_dq10_leakage_field_detected_in_raw(df_raw, schema):
    """DQ-10b: Leakage field (approval_flag) IS present in raw dataset (known defect DFT-009)."""
    leakage_fields = schema.get("leakage_fields", [])
    present = [f for f in leakage_fields if f in df_raw.columns]
    assert len(present) > 0, "Expected leakage field in raw dataset — defect not injected?"


# ---- DQ-11: application_id format ----
def test_dq11_application_id_format(df_clean):
    """DQ-11: application_id must match pattern ^APP\\d+$."""
    pattern = re.compile(r"^APP\d+$")
    invalid = df_clean["application_id"].apply(lambda x: not bool(pattern.match(str(x))))
    assert invalid.sum() == 0, f"Malformed application_ids: {df_clean.loc[invalid, 'application_id'].head().tolist()}"


# ---- DQ-12: Consistency — debt_to_income derived correctly ----
def test_dq12_debt_to_income_consistent(df_clean):
    """DQ-12: debt_to_income must equal existing_debt / annual_income (within tolerance)."""
    mask = df_clean["annual_income"] > 0
    expected = df_clean.loc[mask, "existing_debt"] / df_clean.loc[mask, "annual_income"]
    actual = df_clean.loc[mask, "debt_to_income"]
    diff = (actual - expected).abs()
    inconsistent = (diff > 0.01).sum()
    assert inconsistent == 0, f"{inconsistent} rows have inconsistent debt_to_income"


# ---- DQ-13: Class imbalance assessment ----
def test_dq13_class_imbalance_documented(df_clean):
    """DQ-13: Document class distribution. Flag if high-risk rate is extreme (>85% or <10%)."""
    high_risk_rate = df_clean["synthetic_target"].mean()
    assert 0.10 <= high_risk_rate <= 0.90, (
        f"Extreme class imbalance: high_risk_rate={high_risk_rate:.1%}. "
        "Model training may be unreliable. Investigate or apply class weights."
    )


# ---- DQ-14: Outlier detection — extreme loan amounts ----
def test_dq14_extreme_loan_amount_detected_raw(df_raw):
    """DQ-14: Extreme loan amounts (> £1M) are a red flag — detect in raw dataset."""
    extreme = (df_raw["loan_amount"] > 1_000_000).sum()
    assert extreme > 0, "Expected extreme loan amount in raw dataset — defect not injected?"


# ---- DQ-15: Golden dataset stability ----
def test_dq15_golden_dataset_stable(df_golden):
    """DQ-15: Golden dataset must have exactly 50 rows and no null values in key fields."""
    assert len(df_golden) == 50, f"Golden dataset has {len(df_golden)} rows, expected 50"
    key_fields = ["application_id", "annual_income", "credit_score", "synthetic_target"]
    for f in key_fields:
        nulls = df_golden[f].isna().sum()
        assert nulls == 0, f"Golden dataset has {nulls} nulls in {f}"
