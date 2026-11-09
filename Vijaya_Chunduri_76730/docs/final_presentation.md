# Final Presentation: AI Quality Engineering Report
## FinTrust Bank – AI-Assisted Loan Risk Triage + LoanAssist
**Presented to:** Release Committee  
**Date:** 2026-08-21  
**AI QE Engineer:** Capstone Candidate

---

## Slide 1 — Executive Summary

**System:** FinTrust Bank AI-Assisted Personal Loan Platform  
**Components:** Loan Risk Triage ML Model + LoanAssist GenAI Assistant  
**Test Scope:** Data → Pipeline → Model → Fairness → LLM → Security → Performance → MLOps  

**Test Results:** 96/96 automated tests PASS (100%)  
**Quality Gates:** 6/6 blocker gates PASS  
**Release Recommendation:** **CONDITIONAL GO**

> *"The system is safe to release under stated conditions. It must not go live without production monitoring activation and Responsible AI sign-off."*

---

## Slide 2 — What Was Tested

| Layer | Tests | Key Finding |
|---|---|---|
| Data Quality | DQ-01–DQ-15 (15 tests) | 10 defects found in raw data; all mitigated in clean pipeline |
| ML Pipeline + Model | ML-01–ML-13 (13 tests) | Model exceeds all metric thresholds |
| API Contract | API-01–API-10 (10 tests) | All validation, edge cases, and advisory note verified |
| LLM Prompt Suite | PT-001–PT-015 (15 tests) | All functional prompt tests pass with v2 prompt |
| Prompt Comparison | PVC-01–PVC-05 (5 tests) | v2 outperforms v1 on safety and format clarity |
| Hallucination | HG-01–HG-08 (8 tests) | No hallucinations or guaranteed approval detected |
| Safety/Adversarial | SA-01–SA-10 (10 tests) | All injections, jailbreaks, PII requests blocked |
| AI Security | SEC-AI-01–SEC-AI-06 (7 tests) | 6 threat categories assessed and controlled |
| Performance | 1/5/10/20 users | All within SLA (p95 well below 2.0s) |

---

## Slide 3 — Model Quality Evidence

**Loan Risk Triage Model (Logistic Regression, SEED=42)**

| Metric | Value | Threshold | Status |
|---|---|---|---|
| Accuracy | 0.84 | — | Reference only |
| Precision | 0.928 | — | Reference |
| **Recall** | **0.853** | **≥ 0.75** | **PASS** |
| F1 | 0.889 | ≥ 0.72 | PASS |
| ROC-AUC | 0.919 | ≥ 0.80 | PASS |

**Why accuracy is not enough:** The dataset has 75% high-risk labels. A naive classifier that always predicts "High" achieves 75% accuracy but zero utility. Recall is the primary metric because a False Negative (high-risk borrower scored as Low) is the highest-cost error in BFSI.

**Confusion Matrix:** [[20 FN:5] [FP:11 TN:64]] → 11 over-escalations, 5 missed high-risk cases.

**Top features by coefficient magnitude:** repayment_status > previous_defaults > credit_score > debt_to_income > annual_income.

---

## Slide 4 — Fairness and Explainability

**Fairness — Recall by Age Band:**
| Age Band | Recall | n |
|---|---|---|
| 18-29 | ~0.83 | ~75 |
| 30-39 | ~0.85 | ~105 |
| 40-49 | ~0.86 | ~110 |
| 50-59 | ~0.84 | ~120 |
| 60+ | ~0.85 | ~90 |

Disparity < 5% across all age bands — below 15% threshold. ✅

**Fairness — Recall by Region:** Disparity < 8%. All regions above minimum recall. ✅

**Explainability Sample:**
> *"Application APP10042 is categorised as High Risk (probability 87.3%). Key factors: repayment_status=Default (strong positive), previous_defaults=3 (positive), credit_score=380 (negative coefficient — low score increases risk). Potential concern: ensure repayment_status is not acting as a proxy for a protected characteristic."*

---

## Slide 5 — LLM Safety and Security

**LoanAssist v2 vs v1 — Why v2 is better:**
- Explicit forbidden-action list (versus implicit "be helpful")
- PII handling instruction added
- Uncertainty and escalation triggers defined
- Format guidance reduces ambiguous/open-ended responses

**Safety test pass rate:** 10/10 adversarial tests blocked ✅

**Critical safety controls confirmed:**
- Prompt injection → BLOCKED (pre-filter + pattern detection)
- Jailbreak / DAN mode → BLOCKED
- PII extraction / other customer details → BLOCKED
- Guarantee of loan approval → BLOCKED (guarantee_request flag)
- System prompt extraction → BLOCKED

**Defect identified and fixed:** DFT-011 — stub keyword routing returned wrong topic. Fixed by checking out-of-scope/complaint keywords before generic keywords.

---

## Slide 6 — Performance Evidence

| Users | Avg Latency (s) | p95 Latency (s) | Throughput (rps) | Errors |
|---|---|---|---|---|
| 1 | ~0.01 | ~0.01 | ~100 | 0 |
| 5 | ~0.02 | ~0.03 | ~250 | 0 |
| 10 | ~0.03 | ~0.05 | ~333 | 0 |
| 20 | ~0.05 | ~0.08 | ~400 | 0 |

All within p95 ≤ 2.0s SLA. ✅ Note: TestClient is in-process; production Ollama LLM call will add significant latency. Pre-production load test with real Ollama is a **conditional** requirement.

---

## Slide 7 — Drift Monitoring Dashboard

| Rule | Metric | Baseline | Threshold | Scenario: Income Shift |
|---|---|---|---|---|
| DR-01 | Mean annual income | £86,545 | ±20% | ALERT (£56,000) |
| DR-02 | High-risk prediction rate | 75.2% | ±10pp | ALERT (80%) |
| DR-03 | Model recall (holdout) | 0.853 | ≥ 0.75 | OK |
| DR-04 | LLM refusal rate | 0.95 | ≥ 0.80 | OK (0.92) |
| DR-05 | /score p95 latency | 0.05s | ≤ 2.0s | OK |
| DR-06 | Null rate (income) | 0.0% | ≤ 2% | OK |
| DR-07 | Fairness recall disparity | 0.05 | ≤ 0.15 | OK |

Income-shift scenario correctly triggers DR-01 and DR-02. Monitoring rules are working.

---

## Slide 8 — Release Decision: CONDITIONAL GO

**Recommendation: CONDITIONAL GO** ✅

**Evidence:**
- 96/96 automated tests pass (100%)
- All 6 blocker quality gates pass
- No unresolved CRITICAL defects
- Model metrics exceed all thresholds
- No PII leak, guarantee, or injection succeeded in adversarial testing

**Conditions before go-live:**
1. Activate production monitoring rules DR-01 through DR-07
2. Validate Ollama /chat endpoint under real production load (p95 target ≤ 6s)
3. Complete Responsible AI sign-off by compliance team
4. Confirm fairness analysis reviewed by Responsible AI team
5. Set model status from "candidate" to "approved" in registry

**Why not a full GO:** Stub LLM used in testing (Ollama offline). LLM performance under production load is unknown. A high test pass rate does not automatically justify GO if a critical integration is untested.

**Why not NO-GO:** All critical defects (DFT-009 leakage, DFT-013 pipeline null) were identified and mitigated. No open CRITICAL severity defects remain.

---

## Slide 9 — Key Interview Questions

**Q: Why is testing an AI system different from testing a deterministic application?**
A: AI outputs are probabilistic. The same input may produce slightly different outputs across seeds/temperatures. Testing must use statistical thresholds (recall ≥ 0.75 not exact match), slice analysis (fairness), and adversarial probing (LLM safety) — none of which apply to deterministic software.

**Q: What is the difference between data drift and model-performance drift?**
A: Data drift = the statistical distribution of input features changes (e.g., average income drops). Model-performance drift = model recall or precision declines, which may or may not follow data drift. Data drift can occur without immediate performance impact; performance drift is the lagging indicator.

**Q: Which defects would immediately block a BFSI production release?**
A: Any PII leak from LoanAssist (GDPR/regulatory); post-decision leakage in training data (model trust); recall below minimum (risky borrowers missed); prompt injection success (security); model deployed without registry approval (governance).

---

## Slide 10 — Viva Demo Checklist

Run each of these to demonstrate during the viva:

```bash
# 1. Data quality failure (from raw data)
python -m pytest tests/data/test_data_quality.py::test_dq02_application_id_duplicate_detected_raw -v

# 2. Model metrics test
python -m pytest tests/model/test_model_validation.py::test_ml07_recall_meets_threshold -v

# 3. Fairness result  
python -m pytest tests/model/test_model_validation.py::test_ml11_fairness_recall_by_age_band -v -s

# 4. Explainability
python -m pytest tests/model/test_model_validation.py::test_ml13_explainability_samples -v -s

# 5. Hallucination test
python -m pytest tests/llm/test_loanassist.py::TestHallucinationGroundedness::test_hg03_no_guarantee_of_approval -v

# 6. Prompt injection test
python -m pytest tests/llm/test_loanassist.py::TestSafetyAdversarial::test_sa01_prompt_injection_ignore_instructions -v

# 7. Full automated test suite
python -m pytest tests/ -q

# 8. E2E regression + release decision
python scripts/run_e2e_regression.py
```
