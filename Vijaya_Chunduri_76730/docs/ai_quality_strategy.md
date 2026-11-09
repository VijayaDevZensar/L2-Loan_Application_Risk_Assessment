# AI Quality Strategy
## FinTrust Bank – AI-Assisted Personal Loan Processing and Customer Support
**Version:** 1.0 | **Date:** 2026-08-21 | **Author:** AI QE Capstone Team

---

## 1. Scope and Purpose

This strategy document defines how quality is assured for FinTrust Bank's AI-enabled personal loan platform, which consists of two AI components:

1. **Loan Risk Triage ML Model** — receives applicant data, produces a probability score and Low/Medium/High risk category. **Advisory only.** Final lending decisions remain with authorised bank staff.
2. **LoanAssist GenAI Assistant** — answers customer questions about loan products, eligibility, documents, fees, and process using a retrieval-augmented knowledge base.

The strategy applies the core principle:
> *Do not ask only "Did the feature work?" Ask "Did the system behave acceptably for the intended users, data slices, risk level, and operating conditions — and can we prove it?"*

---

## 2. System Under Test Architecture

```
Applicant/Customer
       │
       ▼
  [Web/API Layer]  ── POST /score ──► [Feature Pipeline] ──► [ML Model] ──► Risk Score
  (FastAPI)        └─ POST /chat ──► [Knowledge Retrieval] ──► [LLM/Ollama] ──► LoanAssist
       │
       ▼
  Loan Officer (human review)  ──► Final Decision
```

**In-scope user journeys:**
- A bank employee submits a synthetic loan application and receives risk triage + explanation.
- A customer asks LoanAssist about products, eligibility, fees, and process.
- A quality engineer validates data, model, prompt, performance, security, and monitoring controls.
- A release pipeline deploys only when quality gates pass.

---

## 3. AI Quality Lifecycle and Gates

| Stage | Activity | Quality Gate |
|---|---|---|
| Data acquisition | Validate schema, completeness, uniqueness | Zero critical schema errors; no leakage fields |
| Data preparation | Clean, feature engineering, class balance | Null rate ≤ 1%; no post-decision leakage |
| Model training | Logistic regression, fixed seed | Reproducibility confirmed; version captured |
| Model validation | Metrics, fairness, explainability | Recall ≥ 0.75; F1 ≥ 0.72; ROC-AUC ≥ 0.80; fairness disparity ≤ 15% |
| GenAI design | Prompt v1/v2, knowledge base | Groundedness/refusal/safety thresholds pass |
| Pre-production | End-to-end, NFR, security | All blocker gates pass |
| Deployment | Registry promotion | Approved version; checksum verified |
| Monitoring | Drift, latency, safety alerts | DR-01–DR-07 rules active |

---

## 4. Test Levels and Coverage

| Level | Tool | Coverage |
|---|---|---|
| Data quality | pytest + pandas | Schema, uniqueness, validity, completeness, leakage (DQ-01–DQ-15) |
| ML pipeline | pytest + sklearn | Train/test separation, imputation, encoding, reproducibility (ML-01–ML-10) |
| Model metrics | pytest + sklearn | Accuracy, precision, recall, F1, ROC-AUC, confusion matrix (ML-07–ML-09) |
| Fairness | pytest + sklearn | Recall and FPR by age band and region (ML-11–ML-12) |
| Explainability | pytest + logistic coef | 3 documented predictions with feature attribution (ML-13) |
| API contract | FastAPI TestClient | Input validation, advisory note, edge cases (API-01–API-10) |
| LLM functional | pytest + stub/Ollama | 15 prompt tests, 5 version comparisons (PT-001–PVC-05) |
| Hallucination | pytest | 8 groundedness/refusal tests (HG-01–HG-08) |
| Safety/adversarial | pytest | 10 injection/PII/jailbreak/role tests (SA-01–SA-10) |
| AI security | pytest | 6 AI-specific security tests (SEC-AI-01–SEC-AI-06) |
| Performance | pytest + threading | p95 latency at 1/5/10/20 users (PERF tests) |
| Drift detection | pytest + drift_monitor | 7 monitoring rules across 4 scenarios |

---

## 5. Test Environments

| Profile | Use | Load | Security |
|---|---|---|---|
| local | Full testing | Enabled | Enabled |
| ci | Automated pipeline | Enabled | Disabled (active attacks off) |
| production | Monitoring only | Synthetic probes | Read-only |

---

## 6. Entry and Exit Criteria

**Entry criteria (per day):**
- Clean dataset available and schema validated
- Model artifacts present (pkl + registry entry)
- API starts successfully
- Knowledge base v1.0 present

**Exit criteria (release):**
- All blocker quality gates: PASS
- Pass rate ≥ 85%
- No unresolved CRITICAL severity defects
- Drift monitoring rules active
- Release recommendation documented and evidence-based

---

## 7. BFSI-Specific Controls

- Model is advisory; human loan officer reviews all AI-assisted scores.
- No protected characteristics (race, religion, gender) in model features.
- All test data is synthetic — no real PII.
- LoanAssist must not guarantee loan outcomes.
- Audit logs must capture model version, input, output, and timestamp for all decisions.
