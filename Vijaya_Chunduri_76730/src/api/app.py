"""
FinTrust Bank AI-Assisted Loan Scoring API.
POST /score  — risk triage (ML model)
POST /chat   — LoanAssist GenAI (Ollama RAG)
GET  /health — liveness
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator

app = FastAPI(
    title="FinTrust Bank AI QE Capstone API",
    description="Loan Risk Triage (ML) + LoanAssist GenAI. Advisory only — not for production lending decisions.",
    version="1.0.0",
)

# ---- Lazy-loaded globals ----
_model = None
_scaler = None
_pipeline = None
_loanassist = None


def _get_model():
    global _model, _scaler, _pipeline
    if _model is None:
        from src.model.train import load_artifacts

        _model, _scaler, _pipeline = load_artifacts()
    return _model, _scaler, _pipeline


def _get_loanassist():
    global _loanassist
    if _loanassist is None:
        from src.llm.loanassist import LoanAssist

        _loanassist = LoanAssist()
    return _loanassist


# ---- Score endpoint ----

ALLOWED_EMPLOYMENT = ["Salaried", "Self-Employed", "Contract", "Retired"]
ALLOWED_REPAYMENT = ["Good", "Delayed", "Default"]
ALLOWED_REGIONS = ["North", "South", "East", "West", "Central"]
ALLOWED_TERMS = [12, 24, 36, 48, 60, 84, 120]


class LoanApplication(BaseModel):
    application_id: str = Field(..., pattern=r"^APP\d+$")
    age: int = Field(..., ge=18, le=100)
    annual_income: float = Field(..., gt=0)
    employment_years: float = Field(..., ge=0)
    existing_debt: float = Field(..., ge=0)
    credit_score: int = Field(..., ge=300, le=850)
    loan_amount: float = Field(..., gt=0)
    loan_term_months: int
    previous_defaults: int = Field(..., ge=0)
    region: str
    employment_type: str
    repayment_status: str

    @field_validator("loan_term_months")
    @classmethod
    def valid_term(cls, v):
        if v not in ALLOWED_TERMS:
            raise ValueError(f"loan_term_months must be one of {ALLOWED_TERMS}")
        return v

    @field_validator("region")
    @classmethod
    def valid_region(cls, v):
        if v not in ALLOWED_REGIONS:
            raise ValueError(f"region must be one of {ALLOWED_REGIONS}")
        return v

    @field_validator("employment_type")
    @classmethod
    def valid_employment(cls, v):
        if v not in ALLOWED_EMPLOYMENT:
            raise ValueError(f"employment_type must be one of {ALLOWED_EMPLOYMENT}")
        return v

    @field_validator("repayment_status")
    @classmethod
    def valid_repayment(cls, v):
        if v not in ALLOWED_REPAYMENT:
            raise ValueError(f"repayment_status must be one of {ALLOWED_REPAYMENT}")
        return v


class ScoreResponse(BaseModel):
    application_id: str
    risk_probability: float
    risk_category: str
    top_factors: list[dict]
    model_version: str
    advisory_note: str


@app.post("/score", response_model=ScoreResponse)
def score_application(app_data: LoanApplication):
    import pandas as pd

    from src.model.train import MODELS_DIR, risk_category

    model, scaler, pipeline = _get_model()

    row = {
        "application_id": app_data.application_id,
        "age": app_data.age,
        "annual_income": app_data.annual_income,
        "employment_years": app_data.employment_years,
        "existing_debt": app_data.existing_debt,
        "credit_score": app_data.credit_score,
        "loan_amount": app_data.loan_amount,
        "loan_term_months": app_data.loan_term_months,
        "previous_defaults": app_data.previous_defaults,
        "region": app_data.region,
        "employment_type": app_data.employment_type,
        "repayment_status": app_data.repayment_status,
        "debt_to_income": app_data.existing_debt / app_data.annual_income,
    }

    df = pd.DataFrame([row])
    try:
        X = pipeline.transform(df)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Feature pipeline error: {e}")

    X_sc = scaler.transform(X)
    prob = float(model.predict_proba(X_sc)[0, 1])
    category = risk_category(prob)

    # Feature contributions
    feature_names = pipeline.feature_names()
    coefs = model.coef_[0]
    contributions = [
        {"feature": n, "coefficient": round(float(c), 4), "value": round(float(X.iloc[0][n]), 4) if n in X.columns else None}
        for n, c in sorted(zip(feature_names, coefs), key=lambda x: abs(x[1]), reverse=True)
    ]

    return ScoreResponse(
        application_id=app_data.application_id,
        risk_probability=round(prob, 4),
        risk_category=category,
        top_factors=contributions[:5],
        model_version="risk_model_v1.0",
        advisory_note="This score is advisory. Final lending decisions are made by authorised FinTrust Bank staff.",
    )


# ---- Chat endpoint ----


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=2000)
    session_id: str = Field(default="anon")
    prompt_version: str = Field(default="v2")


class ChatResponse(BaseModel):
    reply: str
    sources: list[str]
    prompt_version: str
    safety_flags: list[str]
    model_version: str


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    try:
        assistant = _get_loanassist()
        result = assistant.respond(req.message, prompt_version=req.prompt_version)
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"LoanAssist unavailable: {e}")

    return ChatResponse(
        reply=result["reply"],
        sources=result.get("sources", []),
        prompt_version=req.prompt_version,
        safety_flags=result.get("safety_flags", []),
        model_version="llama3",
    )


@app.get("/health")
def health():
    return {"status": "ok", "service": "FinTrust AI QE Capstone", "version": "1.0.0"}
