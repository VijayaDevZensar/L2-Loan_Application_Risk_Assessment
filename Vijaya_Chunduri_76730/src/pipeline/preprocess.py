"""
Feature pipeline: clean → model-ready features.
Same logic must be used at training time and inference time (train/serve parity).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

ROOT = Path(__file__).parent.parent.parent

NUMERIC_FEATURES = [
    "age",
    "annual_income",
    "employment_years",
    "existing_debt",
    "credit_score",
    "loan_amount",
    "loan_term_months",
    "previous_defaults",
    "debt_to_income",
]
CATEGORICAL_FEATURES = ["employment_type", "repayment_status", "region"]

# Allowed categories — mirror schema
ALLOWED_EMPLOYMENT = ["Salaried", "Self-Employed", "Contract", "Retired"]
ALLOWED_REPAYMENT = ["Good", "Delayed", "Default"]
ALLOWED_REGIONS = ["North", "South", "East", "West", "Central"]
UNKNOWN_LABEL = "__unknown__"


class LoanFeaturePipeline:
    """Transforms raw loan application rows into a numeric feature matrix."""

    VERSION = "pipeline_v1.0"

    def __init__(self) -> None:
        self._encoders: dict[str, LabelEncoder] = {}
        self._fitted = False

    def fit(self, df: pd.DataFrame) -> "LoanFeaturePipeline":
        for col, allowed in [
            ("employment_type", ALLOWED_EMPLOYMENT),
            ("repayment_status", ALLOWED_REPAYMENT),
            ("region", ALLOWED_REGIONS),
        ]:
            enc = LabelEncoder()
            enc.fit(allowed + [UNKNOWN_LABEL])
            self._encoders[col] = enc
        self._fitted = True
        return self

    def _safe_encode(self, col: str, series: pd.Series) -> pd.Series:
        enc = self._encoders[col]
        known = set(enc.classes_)
        mapped = series.where(series.isin(known), other=UNKNOWN_LABEL)
        return pd.Series(enc.transform(mapped), index=series.index)

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self._fitted:
            raise RuntimeError("Pipeline not fitted. Call fit() first.")

        result = df.copy()

        # Missing value imputation for numerics (median strategy).
        # Use global fallback medians so single-row inference doesn't yield NaN.
        FALLBACK_MEDIANS = {
            "age": 38,
            "annual_income": 60000.0,
            "employment_years": 5.0,
            "existing_debt": 8000.0,
            "credit_score": 680,
            "loan_amount": 25000.0,
            "loan_term_months": 36,
            "previous_defaults": 0,
            "debt_to_income": 0.15,
        }
        for col in NUMERIC_FEATURES:
            if col in result.columns:
                result[col] = pd.to_numeric(result[col], errors="coerce")
                col_median = result[col].median()
                if pd.isna(col_median):
                    col_median = FALLBACK_MEDIANS.get(col, 0)
                result[col] = result[col].fillna(col_median)

        # Recalculate debt_to_income if missing or derived incorrectly
        if "annual_income" in result.columns and "existing_debt" in result.columns:
            result["debt_to_income"] = np.where(
                result["annual_income"] > 0,
                result["existing_debt"] / result["annual_income"],
                0.0,
            )

        # Categorical encoding
        for col in CATEGORICAL_FEATURES:
            if col in result.columns:
                result[col + "_enc"] = self._safe_encode(col, result[col].astype(str))

        # Drop original categoricals and non-feature columns
        drop_cols = CATEGORICAL_FEATURES + [
            "application_id",
            "age_band",
            "synthetic_target",
            "approval_flag",
            "repayment_status",
        ]
        feature_cols = NUMERIC_FEATURES + [c + "_enc" for c in CATEGORICAL_FEATURES]
        result = result[[c for c in feature_cols if c in result.columns]]

        return result

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)

    def feature_names(self) -> list[str]:
        return NUMERIC_FEATURES + [c + "_enc" for c in CATEGORICAL_FEATURES]

    def save(self, path: Path) -> None:
        import pickle

        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: Path) -> "LoanFeaturePipeline":
        import pickle

        with open(path, "rb") as f:
            return pickle.load(f)
