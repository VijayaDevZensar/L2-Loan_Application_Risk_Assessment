# CI/CD Quality Gates Design
## FinTrust Bank AI QE Capstone
**Version:** 1.0 | **Date:** 2026-08-21

---

## Pipeline Overview

```
Code Commit
    │
    ▼
[1] Data Tests        ── data schema, leakage, uniqueness ──────────► BLOCK on critical failure
    │
    ▼
[2] Unit/API Tests    ── pipeline, API contract, validation ──────────► BLOCK on failure
    │
    ▼
[3] Model Evaluation  ── recall, F1, ROC-AUC, confusion matrix ───────► BLOCK if recall < 0.75
    │
    ▼
[4] Fairness Tests    ── slice recall, disparity check ────────────────► REVIEW/BLOCK if disparity > 15%
    │
    ▼
[5] LLM Safety Tests  ── adversarial, injection, PII ─────────────────► BLOCK on critical PII leak
    │
    ▼
[6] Security Tests    ── API validation, injection detection ──────────► BLOCK on critical failure
    │
    ▼
[7] Performance Tests ── p95 ≤ 2.0s at 10 users ───────────────────────► BLOCK/REVIEW
    │
    ▼
Deployment Decision: PASS → Promote to Staging → Human Responsible AI Sign-off → Production
```

---

## Quality Gates Table

| Gate ID | Stage | Rule | Threshold | Action on Failure | Blocker? | Justification |
|---|---|---|---|---|---|---|
| QG-01 | Data | Critical schema checks pass | 100% conformity | Block deployment | YES | Corrupt data invalidates all downstream results |
| QG-02 | Data | No leakage fields in training data | 0 leakage fields | Block deployment | YES | Post-decision leakage causes model to cheat; regulatory concern |
| QG-03 | Data | Duplicate application_ids | 0 duplicates | Block deployment | YES | Duplicates cause training data bias |
| QG-04 | Model | Recall ≥ threshold | 0.75 | Block deployment | YES | Recall < 0.75 means ≥25% of high-risk borrowers scored as Low Risk — unacceptable BFSI risk |
| QG-05 | Model | F1 ≥ threshold | 0.72 | Block deployment | YES | Ensures balanced precision/recall — not gaming one metric |
| QG-06 | Model | ROC-AUC ≥ threshold | 0.80 | Block deployment | YES | AUC < 0.80 indicates poor discrimination overall |
| QG-07 | Fairness | Recall disparity across age bands | ≤ 15% | Review/Block | REVIEW | Disparity 10–15%: investigate. >15%: block pending explanation |
| QG-08 | Fairness | Recall disparity across regions | ≤ 15% | Review/Block | REVIEW | Same rationale as QG-07 |
| QG-09 | LLM Safety | No critical PII leak in adversarial tests | 0 PII leaks | Block deployment | YES | GDPR violation risk and customer harm |
| QG-10 | LLM Safety | Prompt injection blocked | 100% of injection tests | Block deployment | YES | Security and trust |
| QG-11 | LLM Safety | Guarantee of approval not given | 0 instances | Block deployment | YES | Regulatory and consumer protection |
| QG-12 | Security | API input validation | 100% of validation tests | Block deployment | YES | Malformed inputs could corrupt scores |
| QG-13 | Performance | /score p95 latency | ≤ 2.0s at 10 users | Block/Review | YES | Loan officers need timely responses; SLA breach degrades service |
| QG-14 | Performance | Error rate | ≤ 2% | Block deployment | YES | High error rate indicates instability |

---

## Threshold Justifications

**Recall ≥ 0.75 (QG-04):** In loan risk triage, a False Negative is a high-risk borrower classified as Low Risk. This passes through to a loan officer who may approve a loan that should be escalated. In BFSI, missing risky cases has higher business impact than unnecessary escalations. Recall is therefore the primary model metric. 0.75 is the minimum acceptable floor — the achieved value of 0.853 provides safety margin.

**p95 ≤ 2.0s (QG-13):** Loan officers reviewing applications in real time need sub-second scoring. 2.0s is the outer limit. The achieved p95 of ~0.05s under the test client is well within threshold. Under real production load (FastAPI + Ollama), the limit ensures acceptable UX.

**PII Leaks = 0 (QG-09):** Under UK GDPR, leaking another customer's data via a chat interface is a notifiable data breach. Zero tolerance is the only acceptable threshold.

---

## Automated CI Command

```bash
# Run full CI gate (safe for CI environment — no active security attacks)
python -m pytest tests/data/ tests/model/ tests/api/ tests/llm/ tests/security/ tests/performance/ \
  --tb=short -q

# Run E2E regression and release decision
python scripts/run_e2e_regression.py
```
