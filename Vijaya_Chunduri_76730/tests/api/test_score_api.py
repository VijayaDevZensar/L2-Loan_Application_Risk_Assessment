"""
API CONTRACT AND VALIDATION TESTS — Day 3, Exercise 1 (API layer)
FinTrust Bank AI QE Capstone

Tests the /score and /health endpoints without requiring a running server
by using FastAPI's TestClient.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def client():
    from src.api.app import app
    return TestClient(app)


VALID_APPLICATION = {
    "application_id": "APP12345",
    "age": 35,
    "annual_income": 65000.0,
    "employment_years": 8.0,
    "existing_debt": 10000.0,
    "credit_score": 720,
    "loan_amount": 30000.0,
    "loan_term_months": 36,
    "previous_defaults": 0,
    "region": "North",
    "employment_type": "Salaried",
    "repayment_status": "Good",
}


# ---- API-01: Health endpoint ----
def test_api01_health_endpoint(client):
    """API-01: GET /health returns 200 and status=ok."""
    r = client.get("/health")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "ok"


# ---- API-02: Valid application returns score ----
def test_api02_valid_score_request(client):
    """API-02: Valid application returns risk_probability in [0,1] and valid risk_category."""
    r = client.post("/score", json=VALID_APPLICATION)
    assert r.status_code == 200
    data = r.json()
    assert 0.0 <= data["risk_probability"] <= 1.0
    assert data["risk_category"] in ["Low", "Medium", "High"]
    assert data["application_id"] == "APP12345"
    assert "advisory_note" in data


# ---- API-03: Advisory note is always present ----
def test_api03_advisory_note_present(client):
    """API-03: Response must always include human-advisory disclaimer."""
    r = client.post("/score", json=VALID_APPLICATION)
    assert r.status_code == 200
    advisory = r.json()["advisory_note"].lower()
    assert "advisory" in advisory or "human" in advisory or "staff" in advisory or "decision" in advisory


# ---- API-04: Invalid credit score (too high) ----
def test_api04_invalid_credit_score_rejected(client):
    """API-04: Credit score > 850 must be rejected with 422."""
    bad_app = {**VALID_APPLICATION, "credit_score": 950, "application_id": "APP99001"}
    r = client.post("/score", json=bad_app)
    assert r.status_code == 422, f"Expected 422, got {r.status_code}"


# ---- API-05: Negative income rejected ----
def test_api05_negative_income_rejected(client):
    """API-05: Negative annual_income must be rejected with 422."""
    bad_app = {**VALID_APPLICATION, "annual_income": -10000.0, "application_id": "APP99002"}
    r = client.post("/score", json=bad_app)
    assert r.status_code == 422, f"Expected 422, got {r.status_code}"


# ---- API-06: Invalid employment type rejected ----
def test_api06_invalid_employment_type_rejected(client):
    """API-06: Unsupported employment_type must be rejected with 422."""
    bad_app = {**VALID_APPLICATION, "employment_type": "Freelancer", "application_id": "APP99003"}
    r = client.post("/score", json=bad_app)
    assert r.status_code == 422, f"Expected 422, got {r.status_code}"


# ---- API-07: Invalid loan_term_months rejected ----
def test_api07_invalid_term_rejected(client):
    """API-07: loan_term_months not in allowed set must be rejected with 422."""
    bad_app = {**VALID_APPLICATION, "loan_term_months": 99, "application_id": "APP99004"}
    r = client.post("/score", json=bad_app)
    assert r.status_code == 422, f"Expected 422, got {r.status_code}"


# ---- API-08: Application ID format validation ----
def test_api08_invalid_application_id_rejected(client):
    """API-08: application_id not matching ^APP\\d+$ must be rejected with 422."""
    bad_app = {**VALID_APPLICATION, "application_id": "INVALID-ID"}
    r = client.post("/score", json=bad_app)
    assert r.status_code == 422, f"Expected 422, got {r.status_code}"


# ---- API-09: High-risk application scores high ----
def test_api09_high_risk_application(client):
    """API-09: Application with poor credit profile should produce higher risk score."""
    high_risk_app = {
        "application_id": "APP55555",
        "age": 30,
        "annual_income": 20000.0,
        "employment_years": 1.0,
        "existing_debt": 18000.0,
        "credit_score": 380,
        "loan_amount": 40000.0,
        "loan_term_months": 60,
        "previous_defaults": 3,
        "region": "South",
        "employment_type": "Contract",
        "repayment_status": "Default",
    }
    r = client.post("/score", json=high_risk_app)
    assert r.status_code == 200
    data = r.json()
    assert data["risk_probability"] >= 0.5, (
        f"API-09 FAIL: Expected high probability for poor credit profile, got {data['risk_probability']}"
    )
    assert data["risk_category"] in ["Medium", "High"]


# ---- API-10: Low-risk application scores low ----
def test_api10_low_risk_application(client):
    """API-10: Application with excellent credit profile should produce lower risk score."""
    low_risk_app = {
        "application_id": "APP11111",
        "age": 42,
        "annual_income": 120000.0,
        "employment_years": 15.0,
        "existing_debt": 5000.0,
        "credit_score": 820,
        "loan_amount": 15000.0,
        "loan_term_months": 24,
        "previous_defaults": 0,
        "region": "East",
        "employment_type": "Salaried",
        "repayment_status": "Good",
    }
    r = client.post("/score", json=low_risk_app)
    assert r.status_code == 200
    data = r.json()
    assert data["risk_probability"] < 0.6, (
        f"API-10 FAIL: Expected low probability for excellent credit profile, got {data['risk_probability']}"
    )
