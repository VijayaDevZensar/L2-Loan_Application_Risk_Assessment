# AI Quality Risk Register
## FinTrust Bank – Loan Risk Triage + LoanAssist
**Version:** 1.0 | **Date:** 2026-08-21

| Risk ID | Category | Risk Description | Severity | Likelihood | Mitigation | Quality Gate |
|---|---|---|---|---|---|---|
| RR-01 | Data | Post-decision leakage field (approval_flag) enters model training features | HIGH | HIGH | Remove leakage fields in cleaning pipeline; DQ-10 automated check | Block if leakage field present in training data |
| RR-02 | Data | Duplicate application_ids in training dataset cause biased model | HIGH | MEDIUM | DQ-02 uniqueness check; de-duplication in cleaning pipeline | Block if >0 duplicate IDs in clean data |
| RR-03 | Data | Missing income/credit score values cause silent imputation errors | HIGH | MEDIUM | DQ-03 null check; pipeline imputation with fallback medians | Block if null rate >1% in mandatory fields |
| RR-04 | Model | Model accuracy metric hides class imbalance — high-risk borrowers misclassified | HIGH | HIGH | Use recall, F1, ROC-AUC as primary metrics; apply class_weight='balanced' | Block if recall < 0.75 |
| RR-05 | Fairness | Model produces materially different recall across age bands or regions | HIGH | MEDIUM | Fairness slice analysis (ML-11, ML-12); bias documentation | Block/Review if recall disparity >15% |
| RR-06 | Security | Prompt injection bypasses LoanAssist safety guardrails | HIGH | HIGH | INJECTION_PATTERNS pre-filter; adversarial test suite SA-01–SA-10 | Block if critical injection succeeds |
| RR-07 | Security | LoanAssist reveals another customer's PII or account details | CRITICAL | MEDIUM | PII_REQUEST_PATTERNS filter; no DB access in LoanAssist | Block if PII leak detected |
| RR-08 | Performance | /score API p95 latency exceeds SLA under concurrent load | HIGH | MEDIUM | Performance tests at 1/5/10/20 users; p95 ≤ 2.0s threshold | Block if p95 > 2.0s |
| RR-09 | Fairness | LoanAssist provides discriminatory guidance about eligibility | HIGH | LOW | SA-06 adversarial test; policy.md anti-discrimination rules | Block if discriminatory response detected |
| RR-10 | MLOps | Model deployed without matching feature pipeline version | HIGH | MEDIUM | Model registry with checksum; pipeline versioned alongside model | Block if checksum mismatch |
| RR-11 | Regulatory | Advisory model output treated as autonomous lending decision | CRITICAL | MEDIUM | Advisory note on all /score responses; human review workflow | Block if advisory disclaimer absent |
| RR-12 | Data | Outlier loan amounts (>£1M) corrupt model training distribution | MEDIUM | LOW | DQ-14 outlier check; business-rule bounds in schema | Review if extreme values detected |
| RR-13 | Model | Unknown employment category causes inference failure | HIGH | MEDIUM | __unknown__ label handling in pipeline; ML-03 test | Block if inference crashes on unknown category |
| RR-14 | MLOps | Model/prompt version mismatch in production | HIGH | MEDIUM | Registry entry captures all version dependencies | Block deployment if version manifest incomplete |
| RR-15 | Operational | Data drift causes model recall to degrade post-deployment | HIGH | MEDIUM | DR-03 monitoring rule; weekly holdout evaluation | Alert if recall drops below 0.75 |
