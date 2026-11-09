# FinTrust Bank – End-to-End AI Quality Engineering Capstone

**Candidate:** AI QE Capstone | **Date:** 2026-08-21 | **Level:** Beginner BFSI

A complete AI Quality Engineering capstone for FinTrust Bank's AI-Assisted Personal Loan Platform, implementing the full lifecycle from data validation through ML model testing, LLM safety testing, performance evaluation, drift monitoring, and evidence-based release decision.

**Release Recommendation: CONDITIONAL GO (capstone demonstration only; not production approval)**

---

## Quick Start

```bash
# 1. Create and activate virtual environment
cd Vijaya_Chunduri_AI_QE_BFSI_Capstone
python3 -m venv .venv
source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Build the system under test (generate data + train model)
python scripts/build_sut.py  # validates bundled artifacts; does not retrain

# 4. Run the full test suite
python -m pytest tests/ -v

# 5. Run E2E regression + release decision
python scripts/run_e2e_regression.py
```

---

## System Under Test

Two AI components are tested:

| Component | Description |
|---|---|
| Loan Risk Triage ML | Logistic regression, synthetic loan data, Low/Medium/High risk output |
| LoanAssist GenAI | RAG over approved BFSI knowledge base, Ollama-backed (stub fallback) |

**API endpoints:**
- `POST /score` — risk triage
- `POST /chat` — LoanAssist
- `GET /health` — liveness

---

## Project Structure

```
ai-qe-capstone/
├── data/
│   ├── clean/          loan_data_clean_v1.0.csv (500 rows, no defects)
│   ├── raw/            loan_data_raw_v1.0.csv   (550 rows, 10 defects injected)
│   ├── golden/         golden_50.csv            (50-row regression baseline)
│   └── schemas/        loan_schema_v1.json
├── src/
│   ├── pipeline/       generate_data.py, preprocess.py
│   ├── model/          train.py (loads bundled model artifacts; no retraining)
│   ├── api/            app.py (FastAPI)
│   ├── llm/            loanassist.py (RAG + safety guardrails)
│   └── monitoring/     drift_monitor.py (7 monitoring rules)
├── tests/
│   ├── data/           test_data_quality.py    (15 tests, DQ-01–DQ-15)
│   ├── model/          test_model_validation.py (13 tests, ML-01–ML-13)
│   ├── api/            test_score_api.py       (10 tests, API-01–API-10)
│   ├── llm/            test_loanassist.py      (43 tests: PT, PVC, HG, SA)
│   ├── security/       test_ai_security.py     (7 tests, SEC-AI-01–SEC-AI-06)
│   └── performance/    test_performance.py     (5 tests, 1/5/10/20 users)
├── knowledge/          products.md, policy.md (approved BFSI corpus)
├── prompts/            loanassist_prompt_v1.txt, loanassist_prompt_v2.txt
├── models/             risk_model_v1.pkl, model_registry.json
├── monitoring/         (drift alert configs)
├── reports/
│   ├── day1/           explainability_samples.json
│   ├── day2/           ai_security_risk_register.json, perf_*.json
│   └── day3/           drift_monitoring_*.json, final_release_report.json
├── docs/
│   ├── ai_quality_strategy.md
│   ├── risk_register.md       (15 risks)
│   ├── defect_log.md          (14 defects)
│   ├── versioning_registry.md
│   ├── cicd_quality_gates.md  (14 quality gates)
│   └── final_presentation.md  (10 slides + viva script)
├── scripts/
│   ├── build_sut.py
│   ├── run_e2e_regression.py
│   └── package_submission.py
├── config/             settings.json (thresholds, versions)
└── requirements.txt
```

---

## Test Suite Summary (96 tests)

| Category | Tests | Count |
|---|---|---|
| Data quality (DQ-01–DQ-15) | Schema, uniqueness, validity, leakage, outliers | 15 |
| ML pipeline + model (ML-01–ML-13) | Pipeline, metrics, reproducibility, fairness, explainability | 13 |
| API contract (API-01–API-10) | Validation, advisory note, edge cases | 10 |
| LLM functional (PT-001–PT-015) | Normal, ambiguous, out-of-scope, guarantee | 15 |
| Prompt comparison (PVC-01–PVC-05) | v1 vs v2 on 5 safety-critical cases | 5 |
| Hallucination (HG-01–HG-08) | Non-existent products, guarantees, factual | 8 |
| Safety/adversarial (SA-01–SA-10) | Injection, jailbreak, PII, social engineering | 10 |
| AI security (SEC-AI-01–SEC-AI-06) | API validation, prompt protection, KB integrity | 7 |
| Performance (1/5/10/20 users) | p95 latency, error rate, throughput | 5 |
| E2E regression bonus tests | Via run_e2e_regression.py | — |

**96 tests passed in the local repaired-package run.** See `reports/day3/latest_regression_report.json`.

---

## Key Quality Gates

| Gate | Threshold | Result |
|---|---|---|
| Model Recall | ≥ 0.75 | 0.853 ✅ |
| F1 Score | ≥ 0.72 | 0.889 ✅ |
| ROC-AUC | ≥ 0.80 | 0.919 ✅ |
| Fairness disparity (age) | ≤ 15% | ~4% ✅ |
| LLM Safety (PII) | 0 leaks | 0 leaks ✅ |
| Performance p95 | ≤ 2.0s | ~0.05s ✅ |

---

## Viva Demo Commands

```bash
# Show data defect detection
python -m pytest tests/data/test_data_quality.py::test_dq02_application_id_duplicate_detected_raw -v

# Show model test
python -m pytest tests/model/test_model_validation.py::test_ml07_recall_meets_threshold -v

# Show fairness/explainability
python -m pytest tests/model/test_model_validation.py::test_ml11_fairness_recall_by_age_band -v -s
python -m pytest tests/model/test_model_validation.py::test_ml13_explainability_samples -v -s

# Show LLM hallucination protection
python -m pytest tests/llm/test_loanassist.py::TestHallucinationGroundedness::test_hg03_no_guarantee_of_approval -v

# Show prompt injection blocked
python -m pytest tests/llm/test_loanassist.py::TestSafetyAdversarial::test_sa01_prompt_injection_ignore_instructions -v

# Full automated run
python -m pytest tests/ -q

# Final decision report
python scripts/run_e2e_regression.py
```

---

## Data Policy

All data is **synthetic only**. No real customer PII, account numbers, or production bank data is used at any point. This project is for AI QE training purposes.


## Repair and verification notes (2026-11-06)

The submitted ZIP omitted `config/settings.json`, `src/model/train.py`, and the `scripts` directory. These have been restored. `scripts/build_sut.py` validates the pre-existing bundled model artifacts; it does **not** retrain a model. `scripts/run_e2e_regression.py` writes an actual pytest JUnit XML report and counts all test cases, correcting the historic `reports/day3/final_release_report.json` aggregation issue. Historic reports are retained as archival evidence and must not be mistaken for the current run.

The current suite uses synthetic data and a local offline LoanAssist fallback when Ollama is unavailable. The successful test run is **not** independent verification of a live generative model, bank production integration, or production-scale load. Thresholds in `config/settings.json` are demonstration assumptions. Pickle artifacts originate from the uploaded project and must only be loaded in a trusted environment; scikit-learn version warnings were observed. For an actual bank deployment, obtain a controlled training/reproducibility run, live LLM evaluation, representative load testing, independent fairness assessment, security review and sign-off.

### Reproducible commands

```bash
python -m pip install -r requirements.txt
python scripts/build_sut.py
python scripts/run_e2e_regression.py
python scripts/package_submission.py
```
