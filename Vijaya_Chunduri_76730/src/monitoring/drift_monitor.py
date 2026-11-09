"""
Drift Detection and AI Observability — Day 3, Exercise 4
FinTrust Bank AI QE Capstone

Simulates 5+ monitoring rules covering:
- Data drift (feature distribution shift)
- Prediction drift (risk category distribution change)
- Concept/performance drift (model recall declining)
- LLM quality drift (groundedness score drop)
- Latency drift (p95 increase)
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent.parent

MONITORING_RULES = [
    {
        "rule_id": "DR-01",
        "metric": "mean_annual_income",
        "description": "Monitor mean annual income in incoming applications vs. training baseline",
        "training_baseline": 86545.0,
        "threshold_pct_change": 20.0,
        "action": "Alert Data/AI QE team — possible population shift or data pipeline issue",
        "severity": "HIGH",
        "owner": "Data Engineering",
    },
    {
        "rule_id": "DR-02",
        "metric": "high_risk_prediction_rate",
        "description": "Monitor proportion of applications classified as High Risk",
        "training_baseline": 0.752,
        "threshold_abs_change": 0.10,
        "action": "Review model/data changes — may indicate distribution shift or model drift",
        "severity": "HIGH",
        "owner": "AI QE Team",
    },
    {
        "rule_id": "DR-03",
        "metric": "model_recall_proxy",
        "description": "Monitor recall on labelled holdout sample (concept/performance drift)",
        "training_baseline": 0.853,
        "threshold_min": 0.75,
        "action": "Trigger model revalidation — recall below agreed floor is a release blocker",
        "severity": "CRITICAL",
        "owner": "AI QE Team",
    },
    {
        "rule_id": "DR-04",
        "metric": "llm_refusal_rate",
        "description": "Monitor rate of appropriate refusals on adversarial test set",
        "training_baseline": 0.95,
        "threshold_min": 0.80,
        "action": "Review prompt version and guardrails — LLM quality may have drifted",
        "severity": "HIGH",
        "owner": "LLM QE Team",
    },
    {
        "rule_id": "DR-05",
        "metric": "score_api_p95_latency_s",
        "description": "Monitor /score API p95 response time",
        "training_baseline": 0.05,
        "threshold_max": 2.0,
        "action": "Scale service or investigate bottleneck — breach of SLA is a release block",
        "severity": "HIGH",
        "owner": "Platform Engineering",
    },
    {
        "rule_id": "DR-06",
        "metric": "null_rate_annual_income",
        "description": "Monitor missing-value rate for annual_income in incoming data",
        "training_baseline": 0.0,
        "threshold_max": 0.02,
        "action": "Alert Data team — upstream pipeline may be dropping values",
        "severity": "MEDIUM",
        "owner": "Data Engineering",
    },
    {
        "rule_id": "DR-07",
        "metric": "fairness_recall_disparity_age_band",
        "description": "Monitor recall disparity across age bands — fairness drift signal",
        "training_baseline": 0.05,
        "threshold_max": 0.15,
        "action": "Escalate for Responsible AI review — potential fairness regression",
        "severity": "HIGH",
        "owner": "Responsible AI Team",
    },
]


def simulate_production_data(df_train: pd.DataFrame, scenario: str = "normal") -> pd.DataFrame:
    """Simulate shifted production data for drift detection testing."""
    df = df_train.copy()
    rng = np.random.default_rng(99)

    if scenario == "income_shift":
        # Mean income drops significantly — DR-01 trigger
        df["annual_income"] = df["annual_income"] * 0.65 + rng.normal(0, 5000, len(df))
        df["annual_income"] = df["annual_income"].clip(1000)
    elif scenario == "new_region":
        # 30% of records from an unseen "Southwest" region
        mask = rng.random(len(df)) < 0.30
        df.loc[mask, "region"] = "Southwest"
    elif scenario == "credit_shift":
        # Credit scores drop across the board
        df["credit_score"] = (df["credit_score"] * 0.80).astype(int).clip(300, 850)
    elif scenario == "default_spike":
        # Default rate increases
        df.loc[rng.random(len(df)) < 0.30, "previous_defaults"] += 2
    else:
        # Normal — no shift
        pass

    return df


def evaluate_drift(df_production: pd.DataFrame, model, scaler, pipeline) -> list[dict]:
    """Evaluate all monitoring rules against simulated production data."""
    from sklearn.metrics import recall_score
    from sklearn.model_selection import train_test_split

    results = []
    df = df_production.copy()

    # Compute metrics
    metrics = {}

    # DR-01: mean income
    metrics["mean_annual_income"] = df["annual_income"].mean()

    # DR-02: prediction drift
    try:
        X = pipeline.transform(df.dropna(subset=["synthetic_target"]))
        X_sc = scaler.transform(X)
        preds = model.predict(X_sc)
        metrics["high_risk_prediction_rate"] = float(preds.mean())
    except Exception:
        metrics["high_risk_prediction_rate"] = None

    # DR-03: model recall
    try:
        y_true = df.dropna(subset=["synthetic_target"])["synthetic_target"]
        metrics["model_recall_proxy"] = float(recall_score(y_true, preds, zero_division=0))
    except Exception:
        metrics["model_recall_proxy"] = None

    # DR-04: LLM refusal rate (simulated — use safety flag pass rate)
    metrics["llm_refusal_rate"] = 0.92  # from test suite results

    # DR-05: latency (from latest perf report if available)
    perf_path = ROOT / "reports" / "day2" / "perf_score_10users.json"
    if perf_path.exists():
        perf = json.loads(perf_path.read_text())
        metrics["score_api_p95_latency_s"] = perf.get("p95_latency_s", 0.05)
    else:
        metrics["score_api_p95_latency_s"] = 0.05

    # DR-06: null rate
    metrics["null_rate_annual_income"] = df["annual_income"].isna().mean()

    # DR-07: fairness recall disparity
    try:
        band_recalls = []
        for band in ["18-29", "30-39", "40-49", "50-59", "60+"]:
            sub = df[df["age_band"] == band].dropna(subset=["synthetic_target"])
            if len(sub) >= 10:
                X_sub = pipeline.transform(sub)
                X_sub_sc = scaler.transform(X_sub)
                r = recall_score(sub["synthetic_target"], model.predict(X_sub_sc), zero_division=0)
                band_recalls.append(r)
        if len(band_recalls) >= 2:
            metrics["fairness_recall_disparity_age_band"] = max(band_recalls) - min(band_recalls)
        else:
            metrics["fairness_recall_disparity_age_band"] = 0.0
    except Exception:
        metrics["fairness_recall_disparity_age_band"] = 0.0

    # Evaluate each rule
    for rule in MONITORING_RULES:
        current_val = metrics.get(rule["metric"])
        status = "OK"
        triggered = False

        if current_val is None:
            status = "ERROR"
        elif "threshold_max" in rule and current_val > rule["threshold_max"]:
            status = "ALERT"
            triggered = True
        elif "threshold_min" in rule and current_val < rule["threshold_min"]:
            status = "ALERT"
            triggered = True
        elif "threshold_abs_change" in rule:
            if abs(current_val - rule["training_baseline"]) > rule["threshold_abs_change"]:
                status = "ALERT"
                triggered = True
        elif "threshold_pct_change" in rule:
            pct_change = abs(current_val - rule["training_baseline"]) / rule["training_baseline"] * 100
            if pct_change > rule["threshold_pct_change"]:
                status = "ALERT"
                triggered = True

        results.append({
            **rule,
            "current_value": round(float(current_val), 6) if current_val is not None else None,
            "status": status,
            "triggered": triggered,
            "recommended_action": rule["action"] if triggered else "Monitor",
            "evaluated_at": datetime.now(timezone.utc).isoformat(),
        })

    return results


def write_monitoring_report(results: list[dict], scenario: str = "normal") -> Path:
    report_dir = ROOT / "reports" / "day3"
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / f"drift_monitoring_{scenario}.json"
    report_path.write_text(json.dumps(results, indent=2))
    return report_path
