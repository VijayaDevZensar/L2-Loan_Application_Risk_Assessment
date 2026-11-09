"""
Shared pytest fixtures for FinTrust Bank AI QE Capstone.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

import pandas as pd
import pytest


@pytest.fixture(scope="session")
def root() -> Path:
    return ROOT


@pytest.fixture(scope="session")
def df_clean(root) -> pd.DataFrame:
    path = root / "data" / "clean" / "loan_data_clean_v1.0.csv"
    return pd.read_csv(path)


@pytest.fixture(scope="session")
def df_raw(root) -> pd.DataFrame:
    path = root / "data" / "raw" / "loan_data_raw_v1.0.csv"
    return pd.read_csv(path)


@pytest.fixture(scope="session")
def df_golden(root) -> pd.DataFrame:
    path = root / "data" / "golden" / "golden_50.csv"
    return pd.read_csv(path)


@pytest.fixture(scope="session")
def schema(root) -> dict:
    import json

    return json.loads((root / "data" / "schemas" / "loan_schema_v1.json").read_text())


@pytest.fixture(scope="session")
def model_artifacts(root):
    from src.model.train import load_artifacts

    return load_artifacts()


@pytest.fixture(scope="session")
def pipeline_fitted(model_artifacts, df_clean):
    _, _, pipeline = model_artifacts
    return pipeline


@pytest.fixture(scope="session")
def loanassist():
    from src.llm.loanassist import LoanAssist

    return LoanAssist()
