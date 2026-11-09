"""
ML PIPELINE AND MODEL VALIDATION TESTS — Day 1, Exercise 3 & 4
FinTrust Bank AI QE Capstone

Covers: train/test separation, metrics, fairness, explainability.
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import (
    f1_score,
    recall_score,
    roc_auc_score,
    confusion_matrix,
)

ROOT = Path(__file__).parent.parent.parent
MODELS_DIR = ROOT / "models"
SETTINGS = json.loads((ROOT / "config" / "settings.json").read_text())
THRESHOLDS = SETTINGS["thresholds"]


# ---- ML-01: Train/test split — no leakage between sets ----
def test_ml01_train_test_separation(df_clean, model_artifacts):
    """ML-01: Training data and test set must be disjoint (no application_id overlap)."""
    model, scaler, pipeline = model_artifacts
    from sklearn.model_selection import train_test_split

    X = pipeline.transform(df_clean)
    y = df_clean["synthetic_target"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Index sets are disjoint
    assert len(set(X_train.index) & set(X_test.index)) == 0, "Train and test sets share indices — data leakage!"


# ---- ML-02: Missing value handling in pipeline ----
def test_ml02_pipeline_handles_missing_values(model_artifacts):
    """ML-02: Feature pipeline must not crash when given rows with missing numeric values."""
    model, scaler, pipeline = model_artifacts

    row = {
        "application_id": "APP99999",
        "age": None,
        "annual_income": 60000.0,
        "employment_years": 5.0,
        "existing_debt": 10000.0,
        "credit_score": 700,
        "loan_amount": 25000.0,
        "loan_term_months": 36,
        "previous_defaults": 0,
        "region": "North",
        "employment_type": "Salaried",
        "repayment_status": "Good",
        "debt_to_income": None,
    }
    df = pd.DataFrame([row])
    X = pipeline.transform(df)
    assert X.shape[0] == 1, "Pipeline should return 1 row even with missing values"
    assert not X.isnull().any().any(), "Pipeline output must have no nulls after imputation"


# ---- ML-03: Unknown category handling ----
def test_ml03_unknown_employment_category_handled(model_artifacts):
    """ML-03: Unknown employment_type must be handled gracefully (no crash, no unknown label)."""
    model, scaler, pipeline = model_artifacts

    row = {
        "application_id": "APP88888",
        "age": 35,
        "annual_income": 50000.0,
        "employment_years": 3.0,
        "existing_debt": 5000.0,
        "credit_score": 650,
        "loan_amount": 20000.0,
        "loan_term_months": 36,
        "previous_defaults": 0,
        "region": "East",
        "employment_type": "Freelancer",  # Unknown category
        "repayment_status": "Good",
        "debt_to_income": 0.1,
    }
    df = pd.DataFrame([row])
    X = pipeline.transform(df)
    assert X.shape[0] == 1, "Pipeline must return 1 row for unknown employment type"
    assert not X.isnull().any().any(), "No nulls expected after handling unknown category"


# ---- ML-04: Prediction output format ----
def test_ml04_prediction_format(df_clean, model_artifacts):
    """ML-04: Model must return probability in [0,1] and valid class label."""
    model, scaler, pipeline = model_artifacts
    X = pipeline.transform(df_clean.head(10))
    X_sc = scaler.transform(X)
    probs = model.predict_proba(X_sc)[:, 1]
    preds = model.predict(X_sc)

    assert probs.min() >= 0.0, "Probabilities must be >= 0"
    assert probs.max() <= 1.0, "Probabilities must be <= 1"
    assert set(preds).issubset({0, 1}), f"Unexpected class labels: {set(preds)}"


# ---- ML-05: Model reproducibility ----
def test_ml05_model_reproducible(df_clean, model_artifacts):
    """ML-05: Same inputs must produce identical outputs on repeated calls."""
    model, scaler, pipeline = model_artifacts
    X = pipeline.transform(df_clean.head(5))
    X_sc = scaler.transform(X)

    probs_1 = model.predict_proba(X_sc)[:, 1]
    probs_2 = model.predict_proba(X_sc)[:, 1]

    np.testing.assert_array_equal(probs_1, probs_2, err_msg="Model predictions are not reproducible!")


# ---- ML-06: Invalid input handling (missing required field) ----
def test_ml06_invalid_input_age_string(model_artifacts):
    """ML-06: Pipeline must handle a non-numeric age gracefully (coercion to NaN then imputation)."""
    model, scaler, pipeline = model_artifacts
    row = {
        "application_id": "APP77777",
        "age": "unknown",
        "annual_income": 60000.0,
        "employment_years": 4.0,
        "existing_debt": 8000.0,
        "credit_score": 700,
        "loan_amount": 30000.0,
        "loan_term_months": 48,
        "previous_defaults": 0,
        "region": "West",
        "employment_type": "Contract",
        "repayment_status": "Good",
        "debt_to_income": 0.13,
    }
    df = pd.DataFrame([row])
    X = pipeline.transform(df)
    assert not X["age"].isnull().any(), "Non-numeric age should be imputed, not left as null"


# ---- ML-07: Minimum recall threshold ----
def test_ml07_recall_meets_threshold(df_clean, model_artifacts):
    """ML-07: Model recall must meet minimum threshold. Recall is critical in BFSI — missing high-risk borrowers is costly."""
    model, scaler, pipeline = model_artifacts
    from sklearn.model_selection import train_test_split

    X = pipeline.transform(df_clean)
    y = df_clean["synthetic_target"]
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    X_test_sc = scaler.transform(X_test)
    y_pred = model.predict(X_test_sc)

    recall = recall_score(y_test, y_pred, zero_division=0)
    threshold = THRESHOLDS["recall_min"]
    assert recall >= threshold, (
        f"Recall {recall:.4f} below minimum {threshold}. "
        "Risk: high-risk borrowers classified as Low-risk — RELEASE BLOCKER."
    )


# ---- ML-08: F1 score threshold ----
def test_ml08_f1_meets_threshold(df_clean, model_artifacts):
    """ML-08: Model F1 must meet minimum threshold."""
    model, scaler, pipeline = model_artifacts
    from sklearn.model_selection import train_test_split

    X = pipeline.transform(df_clean)
    y = df_clean["synthetic_target"]
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    X_test_sc = scaler.transform(X_test)
    y_pred = model.predict(X_test_sc)

    f1 = f1_score(y_test, y_pred, zero_division=0)
    threshold = THRESHOLDS["f1_min"]
    assert f1 >= threshold, f"F1 {f1:.4f} below minimum {threshold}"


# ---- ML-09: ROC-AUC threshold ----
def test_ml09_roc_auc_meets_threshold(df_clean, model_artifacts):
    """ML-09: ROC-AUC must meet minimum. Measures discrimination ability across thresholds."""
    model, scaler, pipeline = model_artifacts
    from sklearn.model_selection import train_test_split

    X = pipeline.transform(df_clean)
    y = df_clean["synthetic_target"]
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    X_test_sc = scaler.transform(X_test)
    y_prob = model.predict_proba(X_test_sc)[:, 1]

    auc = roc_auc_score(y_test, y_prob)
    threshold = THRESHOLDS["roc_auc_min"]
    assert auc >= threshold, f"ROC-AUC {auc:.4f} below minimum {threshold}"


# ---- ML-10: Golden dataset regression ----
def test_ml10_golden_dataset_regression(df_golden, model_artifacts):
    """ML-10: Predictions on golden dataset must not change unexpectedly after a model update."""
    model, scaler, pipeline = model_artifacts
    X = pipeline.transform(df_golden)
    X_sc = scaler.transform(X)
    probs = model.predict_proba(X_sc)[:, 1]

    golden_path = ROOT / "data" / "golden" / "golden_predictions_baseline.npy"
    if not golden_path.exists():
        np.save(golden_path, probs)
        pytest.skip("Baseline predictions saved — re-run to compare")

    baseline = np.load(golden_path)
    max_drift = np.abs(probs - baseline).max()
    assert max_drift < 0.05, (
        f"Golden dataset predictions drifted by {max_drift:.4f} — "
        "verify intentional model change and update baseline if approved."
    )


# ---- ML-11: Fairness — recall across age bands ----
def test_ml11_fairness_recall_by_age_band(df_clean, model_artifacts):
    """ML-11: Recall disparity across age bands must not exceed threshold.
    Rationale: If certain age groups have materially lower recall, high-risk applicants in those groups
    are more likely to be misclassified as Low Risk — a BFSI fairness and regulatory concern.
    """
    model, scaler, pipeline = model_artifacts
    disparity_threshold = THRESHOLDS["fairness_recall_disparity_max"]

    results = {}
    for band in ["18-29", "30-39", "40-49", "50-59", "60+"]:
        subset = df_clean[df_clean["age_band"] == band]
        if len(subset) < 10:
            continue
        X = pipeline.transform(subset)
        X_sc = scaler.transform(X)
        y_true = subset["synthetic_target"]
        y_pred = model.predict(X_sc)
        recall = recall_score(y_true, y_pred, zero_division=0)
        results[band] = {"recall": recall, "n": len(subset)}

    if len(results) < 2:
        pytest.skip("Insufficient groups for fairness comparison")

    recalls = [v["recall"] for v in results.values()]
    disparity = max(recalls) - min(recalls)

    report = "\n".join([f"  {band}: recall={v['recall']:.3f} n={v['n']}" for band, v in results.items()])
    print(f"\nFairness — Recall by Age Band:\n{report}\n  Disparity: {disparity:.3f}")

    assert disparity <= disparity_threshold, (
        f"Recall disparity across age bands is {disparity:.3f}, exceeds threshold {disparity_threshold}. "
        f"Investigate for systemic bias.\n{report}"
    )


# ---- ML-12: Fairness — recall across regions ----
def test_ml12_fairness_recall_by_region(df_clean, model_artifacts):
    """ML-12: Recall disparity across synthetic regions must not exceed threshold."""
    model, scaler, pipeline = model_artifacts
    disparity_threshold = THRESHOLDS["fairness_recall_disparity_max"]

    results = {}
    for region in ["North", "South", "East", "West", "Central"]:
        subset = df_clean[df_clean["region"] == region]
        if len(subset) < 10:
            continue
        X = pipeline.transform(subset)
        X_sc = scaler.transform(X)
        y_true = subset["synthetic_target"]
        y_pred = model.predict(X_sc)
        recall = recall_score(y_true, y_pred, zero_division=0)
        results[region] = {"recall": recall, "n": len(subset)}

    if len(results) < 2:
        pytest.skip("Insufficient regions for fairness comparison")

    recalls = [v["recall"] for v in results.values()]
    disparity = max(recalls) - min(recalls)

    report = "\n".join([f"  {r}: recall={v['recall']:.3f} n={v['n']}" for r, v in results.items()])
    print(f"\nFairness — Recall by Region:\n{report}\n  Disparity: {disparity:.3f}")

    assert disparity <= disparity_threshold, (
        f"Recall disparity across regions is {disparity:.3f}, exceeds threshold {disparity_threshold}."
    )


# ---- ML-13: Explainability — 3 predictions documented ----
def test_ml13_explainability_samples(df_clean, model_artifacts):
    """ML-13: For 3 sample predictions, document contributing features and human-readable explanation."""
    model, scaler, pipeline = model_artifacts
    from src.model.train import risk_category

    feature_importance_path = MODELS_DIR / "feature_importance_v1.json"
    assert feature_importance_path.exists(), "Feature importance file not found — run scripts/build_sut.py first"

    fi = json.loads(feature_importance_path.read_text())
    top_features = [item["feature"] for item in fi[:5]]

    samples = df_clean.sample(3, random_state=42)
    explanations = []

    for _, row in samples.iterrows():
        X = pipeline.transform(pd.DataFrame([row]))
        X_sc = scaler.transform(X)
        prob = float(model.predict_proba(X_sc)[0, 1])
        cat = risk_category(prob)

        top_contribs = []
        for feat in top_features:
            if feat in X.columns:
                top_contribs.append(f"{feat}={X.iloc[0][feat]:.2f}")

        explanation = {
            "application_id": row["application_id"],
            "risk_probability": round(prob, 4),
            "risk_category": cat,
            "top_features": top_contribs[:3],
            "human_readable": (
                f"Application {row['application_id']} is categorised as {cat} risk "
                f"(probability {prob:.1%}). Key factors: {', '.join(top_contribs[:3])}."
            ),
            "potential_concern": "Verify no protected attributes are driving the score indirectly via proxy features.",
        }
        explanations.append(explanation)
        print(f"\n{explanation['human_readable']}")

    explainability_path = ROOT / "reports" / "day1" / "explainability_samples.json"
    explainability_path.parent.mkdir(parents=True, exist_ok=True)
    explainability_path.write_text(json.dumps(explanations, indent=2))

    assert len(explanations) == 3, "Must produce explanations for exactly 3 predictions"
    for exp in explanations:
        assert len(exp["top_features"]) > 0, "Each explanation must list at least one contributing feature"
